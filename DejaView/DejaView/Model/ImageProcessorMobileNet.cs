using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Buffers;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Windows.Forms; // for MessageBox

namespace DejaView
{
    internal sealed class ImageProcessorMobileNet : IDisposable
    {
        private const int TargetSize = 400;
        private static readonly float[] Mean = { .485f, .456f, .406f };
        private static readonly float[] Std = { .229f, .224f, .225f };

        private readonly InferenceSession _session;
        private readonly string _inputName;
        private bool _disposed;

        public ImageProcessorMobileNet()
        {
            // Load the embedded ONNX model
            var asm = Assembly.GetExecutingAssembly();
            using Stream? modelStream =
                asm.GetManifestResourceStream("DejaView.Static.mobilenetv2_dynamic.onnx")
                ?? throw new FileNotFoundException("Embedded ONNX model not found.");

            using var ms = new MemoryStream();
            modelStream.CopyTo(ms);
            byte[] modelBytes = ms.ToArray();

            // Keep the CUDA allocator small
            var opts = new SessionOptions
            {
                EnableMemoryPattern = false,
                EnableCpuMemArena = false
            };
            try
            {
                opts.AppendExecutionProvider_CUDA();
            }
            catch (Exception ex) when (
                ex is OnnxRuntimeException
                or DllNotFoundException
                or EntryPointNotFoundException)
            {
                ShowMissingNativeDlls(ex);
                throw;
            }

            _session = new InferenceSession(modelBytes, opts);
            _inputName = _session.InputMetadata.Keys.First();
        }

        public float[][] RunInferenceBatch(IReadOnlyList<byte[]> images)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(ImageProcessorMobileNet));

            int n = images.Count;
            int chw = 3 * TargetSize * TargetSize;

            float[] batched = ArrayPool<float>.Shared.Rent(n * chw);

            try
            {
                Parallel.For(0, n, i =>
                {
                    using var imgMs = new MemoryStream(images[i], writable: false);
                    using var bmp = new Bitmap(imgMs);
                    using var fixedB = ResizeToFixedSize(bmp);

                    float[] norm = NormalizeImage(fixedB);
                    Buffer.BlockCopy(
                        norm, 0,
                        batched, i * chw * sizeof(float),
                        chw * sizeof(float));
                });

                int length = n * chw;
                var memory = new Memory<float>(batched, 0, length);
                var inputTensor = new DenseTensor<float>(
                    memory,
                    new[] { n, 3, TargetSize, TargetSize }
                );

                var inputs = new[] { NamedOnnxValue.CreateFromTensor(_inputName, inputTensor) };

                using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results
                    = _session.Run(inputs);

                var outputTensor = results.First().AsTensor<float>();
                int d = outputTensor.Dimensions[1];

                var flat = outputTensor.ToArray();
                float[][] output = new float[n][];
                for (int i = 0; i < n; i++)
                {
                    output[i] = new float[d];
                    Buffer.BlockCopy(
                        flat, i * d * sizeof(float),
                        output[i], 0, d * sizeof(float));
                }

                return output;
            }
            finally
            {
                ArrayPool<float>.Shared.Return(batched);
            }
        }

        #region Image helpers
        private static Bitmap ResizeToFixedSize(Bitmap src)
        {
            var dst = new Bitmap(TargetSize, TargetSize, PixelFormat.Format24bppRgb);
            using var g = Graphics.FromImage(dst);
            g.InterpolationMode = InterpolationMode.HighQualityBicubic;
            g.Clear(Color.Black);
            g.DrawImage(src, 0, 0, TargetSize, TargetSize);
            return dst;
        }

        private static float[] NormalizeImage(Bitmap img)
        {
            int w = img.Width, h = img.Height;
            float[] data = new float[3 * w * h];

            var bd = img.LockBits(
                new Rectangle(0, 0, w, h),
                ImageLockMode.ReadOnly,
                PixelFormat.Format24bppRgb);

            unsafe
            {
                byte* p = (byte*)bd.Scan0;
                for (int y = 0; y < h; y++)
                {
                    for (int x = 0; x < w; x++)
                    {
                        int idx = y * bd.Stride + x * 3;
                        float b = p[idx] / 255f,
                              g = p[idx + 1] / 255f,
                              r = p[idx + 2] / 255f;
                        int pos = y * w + x;
                        data[pos] = (r - Mean[0]) / Std[0];
                        data[w * h + pos] = (g - Mean[1]) / Std[1];
                        data[2 * w * h + pos] = (b - Mean[2]) / Std[2];
                    }
                }
            }

            img.UnlockBits(bd);
            return data;
        }
        #endregion

        #region IDisposable
        public void Dispose()
        {
            if (_disposed) return;
            _session.Dispose();
            _disposed = true;
        }
        #endregion

        private static void ShowMissingNativeDlls(Exception _)
        {
            MessageBox.Show(
                "Make sure CUDA 12 and cuDNN 9.10 are installed and in your PATH; choose x64 build.",
                "DejaView — Missing Native Dependencies",
                MessageBoxButtons.OK,
                MessageBoxIcon.Error);
        }
    }
}
