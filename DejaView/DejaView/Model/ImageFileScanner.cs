
using System.Collections.Concurrent;
using System.IO;


namespace DejaView.Model
{
    public class RetrievedImagePathsResult
    {
        internal readonly List<string> files;
        internal readonly int nSkippedDirectories;
        internal readonly int nIOExceptions;
        internal RetrievedImagePathsResult(List<string> f, int sd, int io)
        { files = f; nSkippedDirectories = sd; nIOExceptions = io; }
    }
    public class ProcessedImagesResult
    {
        internal readonly Dictionary<string, float[]> embeddings;
        internal readonly int nSkippedImages;
        internal ProcessedImagesResult(Dictionary<string, float[]> e, int s)
        { embeddings = e; nSkippedImages = s; }
    }

    public class ImageFileScanner
    {
        private static readonly Lazy<ImageProcessorMobileNet> _lazyProcessor =
            new(() => new ImageProcessorMobileNet());
        private static ImageProcessorMobileNet Processor => _lazyProcessor.Value;

        public static async Task<RetrievedImagePathsResult> GetAllImagePathsAsync(
            string rootDirectory,
            CancellationToken cancellationToken = default)
        {
            int nSkippedDirectories = 0, nIOExceptions = 0;
            var imageExt = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
                { ".png", ".jpg", ".jpeg" };
            var result = new List<string>();

            if (!Directory.Exists(rootDirectory))
            {
                nIOExceptions++; nSkippedDirectories++;
                return new RetrievedImagePathsResult(result, nSkippedDirectories, nIOExceptions);
            }

            var directories = await Task.Run(() =>
                    Directory.EnumerateDirectories(rootDirectory, "*", SearchOption.AllDirectories)
                             .Prepend(rootDirectory),
                cancellationToken).ConfigureAwait(false);

            foreach (var dir in directories)
            {
                cancellationToken.ThrowIfCancellationRequested();
                try
                {
                    var files = await Task.Run(() => Directory.GetFiles(dir), cancellationToken)
                                           .ConfigureAwait(false);
                    foreach (var file in files)
                    {
                        cancellationToken.ThrowIfCancellationRequested();
                        if (imageExt.Contains(Path.GetExtension(file)))
                            result.Add(file);
                    }
                }
                catch (UnauthorizedAccessException) { nSkippedDirectories++; }
                catch (IOException) { nIOExceptions++; }
            }

            return new RetrievedImagePathsResult(result, nSkippedDirectories, nIOExceptions);
        }

        public static async Task<ProcessedImagesResult> ProcessAllFiles(
            IEnumerable<string> filePaths,
            IProgress<int>? progress = null,
            CancellationToken cancellationToken = default)
        {
            const int BatchSize = 64;
            var paths = filePaths.ToList();
            int total = paths.Count, processed = 0, skipped = 0;
            var embeddings = new ConcurrentDictionary<string, float[]>();

            for (int i = 0; i < total; i += BatchSize)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var batchPaths = paths.Skip(i).Take(BatchSize).ToList();

                // read images concurrently
                var readTasks = batchPaths.Select(async p =>
                {
                    try
                    {
                        var data = await File.ReadAllBytesAsync(p, cancellationToken);
                        return (p, data);
                    }
                    catch
                    {
                        Interlocked.Increment(ref skipped);
                        return (p, (byte[]?)null);
                    }
                });

                var read = await Task.WhenAll(readTasks).ConfigureAwait(false);
                var valid = read.Where(r => r.Item2 != null).ToList();
                if (valid.Count > 0)
                {
                    var imgBytes = valid.Select(v => v.Item2!).ToList();
                    float[][] batchEmb = Processor.RunInferenceBatch(imgBytes);
                    for (int k = 0; k < valid.Count; k++)
                        embeddings[valid[k].p] = batchEmb[k];
                }

                processed += batchPaths.Count;
                int pct = (int)Math.Ceiling((double)processed / total * 100);
                progress?.Report(pct);
            }

            return new ProcessedImagesResult(embeddings.ToDictionary(kv => kv.Key, kv => kv.Value), skipped);
        }
    }
}
