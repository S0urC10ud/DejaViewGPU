using System.Drawing.Imaging;
using System.IO;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Reflection;

namespace DejaView
{
    internal class ImageProcessorSN
    {
        private readonly InferenceSession _session;

        public ImageProcessorSN()
        {
            Assembly assembly = Assembly.GetExecutingAssembly();
            using Stream? stream = assembly.GetManifestResourceStream("DejaView.Static.squeezenet1.1-7.onnx")
                ?? throw new FileNotFoundException($"Embedded squeezenet not found.");

            using MemoryStream ms = new MemoryStream();
            stream.CopyTo(ms);
            byte[] modelBytes = ms.ToArray();

            _session = new InferenceSession(modelBytes);
        }

        public float[] RunInference(byte[] imageBytes)
        {
            using MemoryStream ms = new MemoryStream(imageBytes);
            using Bitmap originalImage = new Bitmap(ms);
            using Bitmap resizedImage = new Bitmap(originalImage, new Size(224, 224));

            float[] imageData = NormalizeImage(resizedImage);
            Tensor<float> inputTensor = CreateTensorFromImage(imageData, resizedImage.Width, resizedImage.Height);

            string inputName = _session.InputMetadata.Keys.First();
            List<NamedOnnxValue> inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor<float>(inputName, inputTensor)
            };

            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = _session.Run(inputs);
            Tensor<float> outputTensor = results.First().AsTensor<float>();
            return outputTensor.ToArray();
        }

        private float[] NormalizeImage(Bitmap image)
        {
            int width = image.Width;
            int height = image.Height;
            // The output array will have three channels (R, G, B) stored in CHW order
            float[] data = new float[3 * width * height];

            float[] mean = { 0.485f, 0.456f, 0.406f };
            float[] std = { 0.229f, 0.224f, 0.225f };

            BitmapData bitmapData = image.LockBits(new Rectangle(0, 0, width, height),
                ImageLockMode.ReadOnly, System.Drawing.Imaging.PixelFormat.Format24bppRgb);

            unsafe
            {
                for (int y = 0; y < height; y++)
                {
                    byte* row = (byte*)bitmapData.Scan0 + (y * bitmapData.Stride);
                    for (int x = 0; x < width; x++)
                    {
                        int pixelIndex = y * width + x;
                        // In Format24bppRgb, the pixel order is Blue, Green, Red
                        float r = row[x * 3 + 2] / 255.0f;
                        float g = row[x * 3 + 1] / 255.0f;
                        float b = row[x * 3 + 0] / 255.0f;
                        // Store in CHW order: channel 0 = red, channel 1 = green, channel 2 = blue
                        data[pixelIndex] = (r - mean[0]) / std[0];
                        data[width * height + pixelIndex] = (g - mean[1]) / std[1];
                        data[2 * width * height + pixelIndex] = (b - mean[2]) / std[2];
                    }
                }
            }

            image.UnlockBits(bitmapData);
            return data;
        }

        private DenseTensor<float> CreateTensorFromImage(float[] imageData, int width, int height)
        {
            DenseTensor<float> tensor = new DenseTensor<float>(new[] { 1, 3, height, width });
            imageData.CopyTo(tensor.Buffer.Span);
            return tensor;
        }
    }
}
