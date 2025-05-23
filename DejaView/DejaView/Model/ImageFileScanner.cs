// ────────────────────────────────────────────────────────────────────────────────
// File: ImageFileScanner.cs   ← no functional change, just a reminder comment
// ────────────────────────────────────────────────────────────────────────────────
using ManagedCuda;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace DejaView.Model
{
    // ↓↓↓ (classes for results are unchanged) ↓↓↓
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
        private static ulong GetFreeGpuMemory()
        {
            try { using var ctx = new CudaContext(); return ctx.GetFreeDeviceMemorySize(); }
            catch (Exception ex)
            {
                MessageBox.Show($"Problem loading CUDA device:\n{ex.Message}",
                                "DejaView — CUDA Device Error",
                                MessageBoxButtons.OK, MessageBoxIcon.Error);
                return 0UL;
            }
        }

        // Any model-load failure now shows *detailed* DLL list first
        private static readonly Lazy<ImageProcessorMobileNet> _lazyProcessor =
            new Lazy<ImageProcessorMobileNet>(() => new ImageProcessorMobileNet());
        private static ImageProcessorMobileNet SharedProcessorMobileNet => _lazyProcessor.Value;


public static async Task<RetrievedImagePathsResult> GetAllImagePathsAsync(
            string rootDirectory,
            CancellationToken cancellationToken = default)
        {
            int nSkippedDirectories = 0, nIOExceptions = 0;
            var imageExtensions = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
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
                        if (imageExtensions.Contains(Path.GetExtension(file)))
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
            var paths = filePaths.ToList();
            int total = paths.Count, processedCount = 0, nSkippedFiles = 0;
            var results = new ConcurrentDictionary<string, float[]>();

            // Estimate image size from a small sample
            int imgWidth = 0, imgHeight = 0;
            int sampleCount = Math.Min(20, paths.Count), totalW = 0, totalH = 0, validCount = 0;
            for (int i = 0; i < sampleCount; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                try
                {
                    var bytes = await File.ReadAllBytesAsync(paths[i], cancellationToken)
                                          .ConfigureAwait(false);
                    using var ms = new MemoryStream(bytes);
                    using var bmp = new Bitmap(ms);
                    using var processed = ImageProcessorMobileNet.PadImageIfNecessary(bmp);
                    totalW += processed.Width;
                    totalH += processed.Height;
                    validCount++;
                }
                catch { }
            }
            if (validCount > 0)
            {
                imgWidth = totalW / validCount;
                imgHeight = totalH / validCount;
            }

            // Compute batch size based on free GPU memory
            int batchSize = 100;
            ulong freeMem = GetFreeGpuMemory();
            if (freeMem > 0 && imgWidth > 0 && imgHeight > 0)
            {
                ulong bytesPerImage = (ulong)(3 * imgWidth * imgHeight * sizeof(float));
                ulong usable = freeMem * 8UL / 10UL;
                batchSize = Math.Max(1, (int)(usable / bytesPerImage));
            }

            // Process files in batches
            for (int i = 0; i < total; i += batchSize)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var batch = paths.Skip(i).Take(batchSize).ToList();

                // Read files concurrently
                var readTasks = batch.Select(async path =>
                {
                    try
                    {
                        var data = await File.ReadAllBytesAsync(path, cancellationToken)
                                             .ConfigureAwait(false);
                        return (path, data);
                    }
                    catch
                    {
                        Interlocked.Increment(ref nSkippedFiles);
                        return (path, (byte[]?)null);
                    }
                });
                var reads = await Task.WhenAll(readTasks).ConfigureAwait(false);

                var valid = reads.Where(r => r.Item2 != null)
                                 .Select(r => (r.path, r.Item2!))
                                 .ToList();
                if (valid.Count > 0)
                {
                    var byteList = valid.Select(v => v.Item2).ToList();
                    var embeddings = SharedProcessorMobileNet.RunInferenceBatch(byteList);
                    for (int j = 0; j < embeddings.Length; j++)
                        results[valid[j].path] = embeddings[j];
                }

                processedCount += batch.Count;
                if (progress != null)
                {
                    int percent = (int)Math.Ceiling((double)processedCount / total * 100);
                    progress.Report(percent);
                }
            }

            return new ProcessedImagesResult(results.ToDictionary(kv => kv.Key, kv => kv.Value),
                                             nSkippedFiles);
        }
    }
}