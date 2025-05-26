using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;
using System.Reflection;

namespace DejaView
{
    internal class ImageProcessorMobileNet : IDisposable
    {
        private const int TargetSize = 800;
        private static readonly float[] Mean = { .485f, .456f, .406f };
        private static readonly float[] Std = { .229f, .224f, .225f };

        private readonly InferenceSession _session;
        private readonly string _inputName;
        private bool _disposed;

        public ImageProcessorMobileNet()
        {
            Assembly asm = Assembly.GetExecutingAssembly();
            using Stream? modelStream =
                asm.GetManifestResourceStream("DejaView.Static.mobilenetv2_dynamic.onnx")
                ?? throw new FileNotFoundException("Embedded ONNX model not found.");

            using var ms = new MemoryStream();
            modelStream.CopyTo(ms);
            byte[] modelBytes = ms.ToArray();

            var opts = new SessionOptions();
            try { opts.AppendExecutionProvider_CUDA(); }
            catch (Exception ex) when (ex is OnnxRuntimeException or DllNotFoundException or EntryPointNotFoundException)
            {
                ShowMissingNativeDlls(ex);
                throw;
            }

            try
            {
                _session = new InferenceSession(modelBytes, opts);
                _inputName = _session.InputMetadata.Keys.First();
            }
            catch (OnnxRuntimeException ortEx)
            {
                ShowMissingNativeDlls(ortEx);
                MessageBox.Show($"ONNX Runtime failed to initialise:\n\n{ortEx.Message}",
                                "DejaView — ONNX Init Error",
                                MessageBoxButtons.OK, MessageBoxIcon.Error);
                throw;
            }
        }

        public float[][] RunInferenceBatch(IReadOnlyList<byte[]> images)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(ImageProcessorMobileNet));

            int n = images.Count;
            int chw = 3 * TargetSize * TargetSize;
            float[] batched = new float[n * chw];

            Parallel.For(0, n, i =>
            {
                using var bmp = new Bitmap(new MemoryStream(images[i]));
                using var fixedB = ResizeToFixedSize(bmp);
                float[] norm = NormalizeImage(fixedB);

                Buffer.BlockCopy(norm, 0, batched, i * chw * sizeof(float), chw * sizeof(float));
            });

            var inputTensor = new DenseTensor<float>(batched, new[] { n, 3, TargetSize, TargetSize });
            var input = DisposableNamedOnnxValue.CreateFromTensor<float>(_inputName, inputTensor);

            using var results = _session.Run(new[] { input });
            var outputTensor = results.First().AsTensor<float>();
            int d = outputTensor.Dimensions[1];

            float[][] output = new float[n][];
            for (int i = 0; i < n; i++)
            {
                output[i] = new float[d];
                var outputTensorArray = outputTensor.ToArray();
                Buffer.BlockCopy(outputTensorArray, i * d * sizeof(float), output[i], 0, d * sizeof(float));
            }
            return output;
        }

        #region Image helpers
        private static Bitmap ResizeToFixedSize(Bitmap src)
        {
            var dst = new Bitmap(TargetSize, TargetSize, PixelFormat.Format24bppRgb);
            using (Graphics g = Graphics.FromImage(dst))
            {
                g.InterpolationMode = InterpolationMode.HighQualityBicubic;
                g.Clear(Color.Black);
                g.DrawImage(src, 0, 0, TargetSize, TargetSize);
            }
            return dst;
        }

        private static float[] NormalizeImage(Bitmap img)
        {
            int w = img.Width, h = img.Height;
            float[] data = new float[3 * w * h];

            BitmapData bd = img.LockBits(new Rectangle(0, 0, w, h),
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
                        data[pos] = (r - Mean[0]) / Std[0];       // R
                        data[w * h + pos] = (g - Mean[1]) / Std[1];       // G
                        data[2 * w * h + pos] = (b - Mean[2]) / Std[2];       // B
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
                "Make sure CUDA 12 and cuDNN 9.10 are installed and their bin directory is in PATH; choose x64 build (ARM won’t work).",
                "DejaView — Missing Native Dependencies",
                MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
    }
}
