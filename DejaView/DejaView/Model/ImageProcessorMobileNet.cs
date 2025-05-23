using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;   // NativeLibrary
using System.Windows.Forms;

namespace DejaView
{
    internal class ImageProcessorMobileNet
    {
        private const int MinSize = 224;
        private readonly InferenceSession _session;

        // DLLs the CUDA provider (ORT 1.22.0 + CUDA 11.6) needs
        private static readonly string[] _expectedNativeDlls =
        {
            "onnxruntime_providers_cuda.dll",
            "cudart64_116.dll",
            "cublas64_11.dll",
            "cublasLt64_11.dll",
            "cudnn64_8.dll",
            "cufft64_10.dll"
        };

        public ImageProcessorMobileNet()
        {
            // 1️⃣ Load embedded ONNX model bytes
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

            // 2️⃣ Build SessionOptions & append CUDA
            var opts = new SessionOptions();
            try
            {
                opts.AppendExecutionProvider_CUDA();
                Console.WriteLine("[DejaView] CUDA execution provider appended.");
            }
            /* ✱✱ NEW ✱✱ — catch OnnxRuntimeException right here */
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

            // 3️⃣ Create the session (rarely reached if CUDA provider failed above)
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

        // ─────────────────────────────────────────────────────────────────────────
        //  Show which DLLs cannot be loaded
        // ─────────────────────────────────────────────────────────────────────────
        private static void ShowMissingNativeDlls(Exception root)
        {
            var missing = new List<string>();
            foreach (string dll in _expectedNativeDlls)
            {
                if (!NativeLibrary.TryLoad(dll, out _))
                    missing.Add(dll);
            }

            string list = missing.Count == 0
                          ? "(none – every DLL found, but one of them failed internally)"
                          : string.Join("\n  • ", missing);

            MessageBox.Show(
                "CUDA execution provider could NOT be loaded.\n\n" +
                "Missing or unloadable DLLs:\n  • " + list +
                "\n\nOriginal error message:\n" + root.Message,
                "DejaView — Missing Native Dependencies",
                MessageBoxButtons.OK, MessageBoxIcon.Error);
        }

        // ─────────────────────────────────────────────────────────────────────────
        //  Rest of the class (unchanged from previous version)
        // ─────────────────────────────────────────────────────────────────────────
        public float[][] RunInferenceBatch(IEnumerable<byte[]> imageBytesList)
        {
            var images = imageBytesList.ToList();
            if (!images.Any()) return Array.Empty<float[]>();

            int batch = images.Count;
            var norm = new List<float[]>();
            int w = 0, h = 0;

            foreach (var bytes in images)
            {
                using var ms = new MemoryStream(bytes);
                using var bmp = new Bitmap(ms);
                using var pad = PadImageIfNecessary(bmp);
                if (w == 0) { w = pad.Width; h = pad.Height; }
                norm.Add(NormalizeImage(pad));
            }

            int len = norm[0].Length;
            var tensor = new DenseTensor<float>(new[] { batch, 3, h, w });
            for (int i = 0; i < batch; i++)
                norm[i].CopyTo(tensor.Buffer.Span.Slice(i * len, len));

            string input = _session.InputMetadata.Keys.First();
            using var res = _session.Run(new[] { NamedOnnxValue.CreateFromTensor(input, tensor) });
            var outT = res.First().AsTensor<float>();
            int feat = outT.Dimensions[1];
            var flat = outT.ToArray();

            var outputs = new float[batch][];
            for (int i = 0; i < batch; i++)
                outputs[i] = flat.Skip(i * feat).Take(feat).ToArray();

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
