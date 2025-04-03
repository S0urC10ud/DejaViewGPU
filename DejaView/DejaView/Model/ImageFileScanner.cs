using System.Collections.Concurrent;
using System.IO;


namespace DejaView.Model
{
    internal class ImageFileScanner
    {
        public static List<string> GetAllImageFiles(string rootDirectory, CancellationToken cancellationToken = default)
        {
            HashSet<string> imageExtensions = new HashSet<string>(StringComparer.OrdinalIgnoreCase) { ".png", ".jpg", ".jpeg" };
            ConcurrentBag<string> result = new ConcurrentBag<string>();

            IEnumerable<string> directories = Directory.EnumerateDirectories(rootDirectory, "*", SearchOption.AllDirectories).Prepend(rootDirectory);

            Parallel.ForEach(directories, new ParallelOptions { CancellationToken = cancellationToken }, dir =>
            {
                try
                {
                    foreach (var file in Directory.EnumerateFiles(dir))
                    {
                        cancellationToken.ThrowIfCancellationRequested();

                        if (imageExtensions.Contains(Path.GetExtension(file)))
                        {
                            result.Add(file);
                        }
                    }
                }
                catch (UnauthorizedAccessException) { /* Skip directories without access */ }
                catch (IOException) { /* Handle or log if needed */ }
            });

            return result.ToList();
        }
        public static async Task<Dictionary<string, float[]>> ProcessAllFilesAsync(
            IEnumerable<string> filePaths,
            IProgress<int>? progress = null,
            CancellationToken cancellationToken = default)
        {
            int processedCount = 0;
            // Limit the concurrency to a reasonable number (number of logical processors, accounts for SMT).
            int maxDegreeOfParallelism = Environment.ProcessorCount;
            SemaphoreSlim semaphore = new SemaphoreSlim(maxDegreeOfParallelism);
            ConcurrentDictionary<string, float[]> results = new ConcurrentDictionary<string, float[]>();

            IEnumerable<Task> tasks = filePaths.Select(async file =>
            {
                // Wait until we can start another file read
                await semaphore.WaitAsync(cancellationToken);
                try
                {
                    // Read file content asynchronously without blocking a thread
                    byte[] content = await File.ReadAllBytesAsync(file, cancellationToken);
                    results[file] = GetVectorRepresentation(content);
                }
                catch (Exception ex)
                {
                    // Optionally log or handle individual file errors here
                    Console.Error.WriteLine($"Error reading file '{file}': {ex.Message}");
                }
                finally
                {
                    semaphore.Release();
                    if (progress != null)
                    {
                        int newCount = Interlocked.Increment(ref processedCount); // increment out of semaphore
                        progress.Report(newCount);
                    }
                }
            });

            // Await all file-read tasks to complete
            await Task.WhenAll(tasks);
            return results.ToDictionary(kvp => kvp.Key, kvp => kvp.Value);
        }

        private static float[] GetVectorRepresentation(byte[] content)
        {
            throw new NotImplementedException();
        }
    }
}
