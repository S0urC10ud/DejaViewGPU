using System.Collections.Concurrent;
using System.IO;


namespace DejaView.Model
{
    internal class RetrievedImagePathsResult
    {
        internal readonly List<string> files;
        internal readonly int nSkippedDirectories;
        internal readonly int nIOExceptions;

        internal RetrievedImagePathsResult(List<string> files, int nSkippedDirectories, int nIOExceptions)
        {
            this.files = files;
            this.nSkippedDirectories = nSkippedDirectories;
            this.nIOExceptions = nIOExceptions;
        }
    }
    internal class ProcessedImagesResult
    {
        internal readonly Dictionary<string, float[]> embeddings;
        internal readonly int nSkippedImages;

        internal ProcessedImagesResult(Dictionary<string, float[]> embeddings, int nSkippedDFiles)
        {
            this.embeddings = embeddings;
            this.nSkippedImages = nSkippedDFiles;
        }
    }

    internal class ImageFileScanner
    {
        // Loads the model into memory only once (ONNX is thread-safe)
        private static readonly ImageProcessorMobileNet SharedProcessorMobileNet = new ImageProcessorMobileNet();

        // The following shared processor can be used as a drop-in replacement which runs a bit faster but has a lower quality in the embedding space:
        // private static readonly ImageProcessorSN SharedProcessorSN = new ImageProcessorSN();

        public static async Task<RetrievedImagePathsResult> GetAllImageFilesAsync(string rootDirectory, CancellationToken cancellationToken = default)
        {
            int nSkippedDirectories = 0;
            int nIOExceptions = 0;

            HashSet<string> imageExtensions = new HashSet<string>(StringComparer.OrdinalIgnoreCase) { ".png", ".jpg", ".jpeg" };

            // Use a thread-safe bag (as there is no concurrent set in .NET) to accumulate results
            ConcurrentBag<string> result = new ConcurrentBag<string>();

            IEnumerable<string> directories = Directory
                .EnumerateDirectories(rootDirectory, "*", SearchOption.AllDirectories)
                .Prepend(rootDirectory);

            // Suitable for I/O-bound or async operations
            await Parallel.ForEachAsync(
                directories,
                new ParallelOptions { CancellationToken = cancellationToken },
                async (dir, token) =>
                {
                    try
                    {
                        // Wrap synchronous file enumeration in Task.Run to prevent blocking
                        string[] files = await Task.Run(() => Directory.GetFiles(dir), token);
                        foreach (var file in files)
                        {
                            token.ThrowIfCancellationRequested();

                            if (imageExtensions.Contains(Path.GetExtension(file)))
                            {
                                result.Add(file);
                            }
                        }
                    }
                    catch (UnauthorizedAccessException)
                    {
                        // Skip directories that cannot be accessed
                        Interlocked.Increment(ref nSkippedDirectories);
                    }
                    catch (IOException)
                    {
                        // I/O exceptions are ignored
                        Interlocked.Increment(ref nIOExceptions);
                    }
                });

            return new RetrievedImagePathsResult(result.ToList(), nSkippedDirectories, nIOExceptions);
        }

        public static async Task<ProcessedImagesResult> ProcessAllFilesAsync(
            IEnumerable<string> filePaths,
            IProgress<int>? progress = null,
            CancellationToken cancellationToken = default)
        {
            int processedCount = 0;
            int nSkippedFiles = 0;
            int nFilePaths = filePaths.Count();
            int maxDegreeOfParallelism = Math.Max(1, Environment.ProcessorCount / 2); // Reduce load
            ConcurrentDictionary<string, float[]> results = new ConcurrentDictionary<string, float[]>();

            // Todo: consider non-offloading async against threadpool-starvation if longrunning is not used https://chatgpt.com/c/67f39d44-c62c-800f-b2a2-326aa1e71f51
            await Parallel.ForEachAsync(
                filePaths,
                new ParallelOptions
                {
                    MaxDegreeOfParallelism = maxDegreeOfParallelism,
                    CancellationToken = cancellationToken
                },
                async (file, token) =>
                {
                    try
                    {
                        byte[] content = await File.ReadAllBytesAsync(file, token);
                        // TODO: Benchmark vs non-await and keeping it in the Threadpool
                        results[file] = await Task.Factory.StartNew(
                            () => SharedProcessorMobileNet.RunInference(content),
                            token,
                            TaskCreationOptions.LongRunning, // Creates a new Thread as RunInference is CPU-heavy
                            TaskScheduler.Default
                        );
                    }
                    catch (Exception)
                    {
                        Interlocked.Increment(ref nSkippedFiles);
                    }
                    finally
                    {
                        if (progress != null)
                        {
                            int newCount = Interlocked.Increment(ref processedCount);
                            progress.Report((int) Math.Ceiling(((double) newCount) / nFilePaths * 100));
                        }
                    }
                });

            return new ProcessedImagesResult(results.ToDictionary(kvp => kvp.Key, kvp => kvp.Value), nSkippedFiles);
        }
    }
}
