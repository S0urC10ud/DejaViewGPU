using System.Collections.Concurrent;
using System.IO;


namespace DejaView.Model
{
    internal class ImageFileScanner
    {
        // Loads the model into memory only once (ONNX is thread-safe)
        private static readonly ImageProcessorSN SharedProcessorSN = new ImageProcessorSN();
        private static readonly ImageProcessorMobileNet SharedProcessorMobileNet = new ImageProcessorMobileNet();

        public static async Task<List<string>> GetAllImageFilesAsync(string rootDirectory, CancellationToken cancellationToken = default)
        {
            var imageExtensions = new HashSet<string>(StringComparer.OrdinalIgnoreCase) { ".png", ".jpg", ".jpeg" };

            // Use a thread-safe bag to accumulate results.
            var result = new ConcurrentBag<string>();

            IEnumerable<string> directories = Directory
                .EnumerateDirectories(rootDirectory, "*", SearchOption.AllDirectories)
                .Prepend(rootDirectory);

            // (available in .NET 6+)
            await Parallel.ForEachAsync(
                directories,
                new ParallelOptions { CancellationToken = cancellationToken },
                async (dir, token) =>
                {
                    try
                    {
                        // Wrap synchronous file enumeration in Task.Run to prevent blocking.
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
                        // Skip directories that cannot be accessed.
                    }
                    catch (IOException)
                    {
                        // I/O exceptions are ignored.
                    }
                });

            return result.ToList();
        }

        public static async Task<Dictionary<string, float[]>> ProcessAllFilesAsync(
            IEnumerable<string> filePaths,
            IProgress<int>? progress = null,
            CancellationToken cancellationToken = default)
        {
            int processedCount = 0;
            int maxDegreeOfParallelism = Math.Max(1, Environment.ProcessorCount / 2); // Reduce load
            ConcurrentDictionary<string, float[]> results = new ConcurrentDictionary<string, float[]>();

            await Parallel.ForEachAsync(
                filePaths,
                new ParallelOptions
                {
                    MaxDegreeOfParallelism = maxDegreeOfParallelism,
                    CancellationToken = cancellationToken
                },
                async (file, ct) =>
                {
                    try
                    {
                        byte[] content = await File.ReadAllBytesAsync(file, ct);
                        // TODO: Benchmark vs non-await and keeping it in the Threadpool
                        results[file] = await Task.Factory.StartNew(
                            () => SharedProcessorMobileNet.RunInference(content),
                            ct,
                            TaskCreationOptions.LongRunning, // Creates a new Thread as RunInference is CPU-heavy
                            TaskScheduler.Default
                        );
                    }
                    catch (Exception ex)
                    {
                        Console.Error.WriteLine($"Error reading or processing file '{file}': {ex.Message}");
                    }
                    finally
                    {
                        if (progress != null)
                        {
                            int newCount = Interlocked.Increment(ref processedCount);
                            System.Diagnostics.Debug.WriteLine($"Processed file {newCount}");
                            progress.Report(newCount);
                        }
                    }
                });

            return results.ToDictionary(kvp => kvp.Key, kvp => kvp.Value);
        }
    }
}
