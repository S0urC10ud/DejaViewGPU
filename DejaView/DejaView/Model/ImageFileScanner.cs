using System.Collections.Concurrent;
using System.IO;


namespace DejaView.Model
{
    public class RetrievedImagePathsResult
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

    public class ProcessedImagesResult
    {
        internal readonly Dictionary<string, float[]> embeddings;
        internal readonly int nSkippedImages;

        internal ProcessedImagesResult(Dictionary<string, float[]> embeddings, int nSkippedImages)
        {
            this.embeddings = embeddings;
            this.nSkippedImages = nSkippedImages;
        }
    }


    public class ImageFileScanner
    {
        // Loads the model into memory only once (ONNX is thread-safe)
        private static readonly ImageProcessorMobileNet SharedProcessorMobileNet = new ImageProcessorMobileNet();

        // The following shared processor can be used as a drop-in replacement which runs a bit faster but has a lower-quality embedding space:
        // private static readonly ImageProcessorSN SharedProcessorSN = new ImageProcessorSN();

        public static async Task<RetrievedImagePathsResult> GetAllImagePathsAsync(string rootDirectory, CancellationToken cancellationToken = default)
        {
            int nSkippedDirectories = 0;
            int nIOExceptions = 0;
            HashSet<string> imageExtensions = new HashSet<string>(StringComparer.OrdinalIgnoreCase) { ".png", ".jpg", ".jpeg" };

            // Since directory enumeration is fast, use a simple list to collect file paths
            List<string> result = new List<string>();

            if (!Directory.Exists(rootDirectory))
            {
                nIOExceptions++; // Count it as an I/O issue
                nSkippedDirectories++;
                return new RetrievedImagePathsResult(result, nSkippedDirectories, nIOExceptions);
            }

            // ConfigureAwait(false): don't necessarily come back to UI Thread
            IEnumerable<string> directories = await Task.Run(() =>
                        Directory.EnumerateDirectories(rootDirectory, "*", SearchOption.AllDirectories)
                                                        .Prepend(rootDirectory), cancellationToken).ConfigureAwait(false);

            // Process each directory sequentially in an async loop.
            foreach (string dir in directories)
            {
                cancellationToken.ThrowIfCancellationRequested();
                try
                {
                    string[] files = await Task.Run(() => Directory.GetFiles(dir), cancellationToken)
                        .ConfigureAwait(false);

                    foreach (string file in files)
                    {
                        cancellationToken.ThrowIfCancellationRequested();

                        if (imageExtensions.Contains(Path.GetExtension(file)))
                            result.Add(file);
                    }
                }
                catch (UnauthorizedAccessException)
                {
                    // Count directories we were not permitted to access
                    nSkippedDirectories++;
                }
                catch (IOException)
                {
                    // Count directories that encountered an I/O exception
                    nIOExceptions++;
                }
            }

            return new RetrievedImagePathsResult(result, nSkippedDirectories, nIOExceptions);
        }


        public static async Task<ProcessedImagesResult> ProcessAllFilesLongRunningForEachAsync(
            IEnumerable<string> filePaths,
            IProgress<int>? progress = null,
            CancellationToken cancellationToken = default)
        {
            int processedCount = 0;
            int nSkippedFiles = 0;
            int nFilePaths = filePaths.Count();

            int maxDegreeOfParallelism = Math.Max(1, Environment.ProcessorCount / 2); // Reduce load
            ConcurrentDictionary<string, float[]> results = new ConcurrentDictionary<string, float[]>();

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

                    if (progress != null)
                    {
                        int newCount = Interlocked.Increment(ref processedCount);
                        progress.Report((int)Math.Ceiling(((double)newCount) / nFilePaths * 100));
                    }
                });

            return new ProcessedImagesResult(results.ToDictionary(kvp => kvp.Key, kvp => kvp.Value), nSkippedFiles);
        }

        // Less-efficient alternative to ProcessAllFilesLongRunningForEachAsync
        public static async Task<ProcessedImagesResult> ProcessAllFilesNoLongRunningForEachAsync(
            IEnumerable<string> filePaths,
            IProgress<int>? progress = null,
            CancellationToken cancellationToken = default)
        {
            int processedCount = 0;
            int nSkippedFiles = 0;
            int nFilePaths = filePaths.Count();

            int maxDegreeOfParallelism = Math.Max(1, Environment.ProcessorCount / 2);
            ConcurrentDictionary<string, float[]> results = new ConcurrentDictionary<string, float[]>();

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
                        results[file] = await Task.Run(() => SharedProcessorMobileNet.RunInference(content), token);
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
                            progress.Report((int)Math.Ceiling(((double)newCount) / nFilePaths * 100));
                        }
                    }
                });

            return new ProcessedImagesResult(results.ToDictionary(kvp => kvp.Key, kvp => kvp.Value), nSkippedFiles);
        }

        // Less-efficient alternative to ProcessAllFilesLongRunningForEachAsync
        public static Task<ProcessedImagesResult> ProcessAllFilesNoLongRunningForEach(
            IEnumerable<string> filePaths,
            IProgress<int>? progress = null,
            CancellationToken cancellationToken = default)
        {
            int processedCount = 0;
            int nSkippedFiles = 0;
            int nFilePaths = filePaths.Count();
            int maxDegreeOfParallelism = Math.Max(1, Environment.ProcessorCount / 2);
            ConcurrentDictionary<string, float[]> results = new ConcurrentDictionary<string, float[]>();

            Parallel.ForEach(filePaths, new ParallelOptions
            {
                MaxDegreeOfParallelism = maxDegreeOfParallelism,
                CancellationToken = cancellationToken
            },
            file =>
            {
                try
                {
                    byte[] content = File.ReadAllBytesAsync(file, cancellationToken).GetAwaiter().GetResult();
                    results[file] = Task.Run(() => SharedProcessorMobileNet.RunInference(content), cancellationToken)
                                        .GetAwaiter().GetResult();
                }
                catch (Exception)
                {
                    Interlocked.Increment(ref nSkippedFiles);
                }
                finally
                {
                    int newCount = Interlocked.Increment(ref processedCount);
                    progress?.Report((int)Math.Ceiling((double)newCount / nFilePaths * 100));
                }
            });

            return Task.FromResult(new ProcessedImagesResult(results.ToDictionary(kvp => kvp.Key, kvp => kvp.Value), nSkippedFiles));
        }

        // Less-efficient alternative to ProcessAllFilesLongRunningForEachAsync
        public static Task<ProcessedImagesResult> ProcessAllFilesLongRunningForEach(
            IEnumerable<string> filePaths,
            IProgress<int>? progress = null,
            CancellationToken cancellationToken = default)
        {
            int processedCount = 0;
            int nSkippedFiles = 0;
            int nFilePaths = filePaths.Count();
            int maxDegreeOfParallelism = Math.Max(1, Environment.ProcessorCount / 2);
            ConcurrentDictionary<string, float[]> results = new ConcurrentDictionary<string, float[]>();

            Parallel.ForEach(filePaths, new ParallelOptions
            {
                MaxDegreeOfParallelism = maxDegreeOfParallelism,
                CancellationToken = cancellationToken
            },
            file =>
            {
                try
                {
                    byte[] content = File.ReadAllBytesAsync(file, cancellationToken).GetAwaiter().GetResult();
                    results[file] = Task.Factory.StartNew(
                            () => SharedProcessorMobileNet.RunInference(content),
                            cancellationToken,
                            TaskCreationOptions.LongRunning,
                            TaskScheduler.Default
                        ).GetAwaiter().GetResult();
                }
                catch (Exception)
                {
                    Interlocked.Increment(ref nSkippedFiles);
                }
                finally
                {
                    int newCount = Interlocked.Increment(ref processedCount);
                    progress?.Report((int)Math.Ceiling((double)newCount / nFilePaths * 100));
                }
            });

            return Task.FromResult(new ProcessedImagesResult(results.ToDictionary(kvp => kvp.Key, kvp => kvp.Value), nSkippedFiles));
        }
    }
}
