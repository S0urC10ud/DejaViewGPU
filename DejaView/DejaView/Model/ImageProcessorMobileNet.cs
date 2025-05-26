using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Drawing.Imaging;
using System.IO;
using System.Reflection;

namespace DejaView
{
    internal class ImageProcessorMobileNet
    {
        private const int MinSize = 224;
        private readonly InferenceSession _session;

        public ImageProcessorMobileNet()
        {
            Assembly asm = Assembly.GetExecutingAssembly();
            using Stream? modelStream =
                asm.GetManifestResourceStream("DejaView.Static.mobilenetv2_dynamic.onnx")
                ?? throw new FileNotFoundException("Embedded ONNX model not found.");

            byte[] modelBytes;
            using (var ms = new MemoryStream())
            {
                modelStream.CopyTo(ms);
                modelBytes = ms.ToArray();
            }

            var opts = new SessionOptions();
            try
            {
                opts.AppendExecutionProvider_CUDA();
                Console.WriteLine("[DejaView] CUDA execution provider appended.");
            }
            catch (OnnxRuntimeException ortEx)
            {
                ShowMissingNativeDlls(ortEx);
                throw;   // still bubble up
            }
            catch (DllNotFoundException dllEx)
            {
                ShowMissingNativeDlls(dllEx);
                throw;
            }
            catch (EntryPointNotFoundException epEx)
            {
                ShowMissingNativeDlls(epEx);
                throw;
            }
            try
            {
                _session = new InferenceSession(modelBytes, opts);
            }
            catch (OnnxRuntimeException ortEx)
            {
                ShowMissingNativeDlls(ortEx);
                MessageBox.Show(
                    $"ONNX Runtime failed to initialise:\n\n{ortEx.Message}",
                    "DejaView — ONNX Init Error",
                    MessageBoxButtons.OK, MessageBoxIcon.Error);
                throw;
            }
        }
        private static void ShowMissingNativeDlls(Exception root)
        {
            MessageBox.Show(
                "Make sure CUDA 12 and cuDNN 9.10 are installed and their bin directory is in PATH; If you build from source, please choose the build option x64 instead of Any (ARM wont work)",
                "DejaView — Missing Native Dependencies",
                MessageBoxButtons.OK, MessageBoxIcon.Error);
        }

        public float[][] RunInferenceBatch(IEnumerable<byte[]> imageBytesList)
        {
            // 1) Decode & pad small images up to MinSize×MinSize
            var padded = imageBytesList.Select(bytes =>
            {
                using var ms = new MemoryStream(bytes);
                using var bmp = new Bitmap(ms);
                return PadImageIfNecessary(bmp);     // returns *new* Bitmap we own
            }).ToList();

            if (padded.Count == 0) return Array.Empty<float[]>();
            int batch = padded.Count;

            // 2) Find the largest W/H across the batch
            int targetW = padded.Max(b => b.Width);
            int targetH = padded.Max(b => b.Height);
            int sliceLen = 3 * targetW * targetH;    // length of one sample after normalisation

            // 3) Letter-box every bitmap into exactly targetW × targetH
            var uniform = padded.Select(src =>
            {
                var canvas = new Bitmap(targetW, targetH, PixelFormat.Format24bppRgb);
                using var g = Graphics.FromImage(canvas);
                g.Clear(Color.White);

                int x = (targetW - src.Width) / 2;
                int y = (targetH - src.Height) / 2;
                g.DrawImage(src, x, y, src.Width, src.Height);

                src.Dispose();                      // free the intermediate padded bitmap
                return canvas;                      // caller disposes *this* one after use
            }).ToList();

            // 4) Normalise → float[] and build tensor
            var tensor = new DenseTensor<float>(new[] { batch, 3, targetH, targetW });
            var span = tensor.Buffer.Span;

            for (int i = 0; i < batch; i++)
            {
                float[] norm = NormalizeImage(uniform[i]);   // returns 3*W*H floats
                norm.CopyTo(span.Slice(i * sliceLen, sliceLen));
                uniform[i].Dispose();                        // no longer needed
            }

            // 5) Run the model
            string inputName = _session.InputMetadata.Keys.First();
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results =
                _session.Run(new[] { NamedOnnxValue.CreateFromTensor(inputName, tensor) });

            var outTensor = results.First().AsTensor<float>();
            int features = outTensor.Dimensions[1];
            var flat = outTensor.ToArray();

            // 6) Split the flat output into per-image arrays
            var outputs = new float[batch][];
            for (int i = 0; i < batch; i++)
                outputs[i] = flat.Skip(i * features).Take(features).ToArray();

            return outputs;
        }


        public static Bitmap PadImageIfNecessary(Bitmap img)
        {
            if (img.Width >= MinSize && img.Height >= MinSize)
                return new Bitmap(img);

            int w = Math.Max(img.Width, MinSize), h = Math.Max(img.Height, MinSize);
            var padded = new Bitmap(w, h);
            using var g = Graphics.FromImage(padded);
            g.Clear(Color.White);
            g.DrawImage(img, (w - img.Width) / 2, (h - img.Height) / 2);
            return padded;
        }

        private float[] NormalizeImage(Bitmap img)
        {
            int w = img.Width, h = img.Height;
            float[] data = new float[3 * w * h];
            float[] mean = { .485f, .456f, .406f };
            float[] std = { .229f, .224f, .225f };

            BitmapData bd = img.LockBits(new Rectangle(0, 0, w, h),
                                         ImageLockMode.ReadOnly,
                                         PixelFormat.Format24bppRgb);
            unsafe
            {
                byte* p = (byte*)bd.Scan0;
                for (int y = 0; y < h; y++)
                    for (int x = 0; x < w; x++)
                    {
                        int idx = y * bd.Stride + x * 3;
                        float b = p[idx] / 255f,
                              g = p[idx + 1] / 255f,
                              r = p[idx + 2] / 255f;
                        int pos = y * w + x;
                        data[pos] = (r - mean[0]) / std[0];
                        data[w * h + pos] = (g - mean[1]) / std[1];
                        data[2 * w * h + pos] = (b - mean[2]) / std[2];
                    }
            }
            img.UnlockBits(bd);
            return data;
        }
    }
}
