using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Reflection;

namespace DejaView
{
    internal class ImageProcessorMobileNet
    {
        private const int MinSize = 224;
        private readonly InferenceSession _session;

        public ImageProcessorMobileNet()
        {
            Assembly assembly = Assembly.GetExecutingAssembly();
            // Note that this model was custom-exported from pytorch (create_model.py)
            using Stream? stream = assembly.GetManifestResourceStream("DejaView.Static.mobilenetv2_dynamic.onnx")
                ?? throw new FileNotFoundException("Embedded MobileNet model not found.");

            using MemoryStream ms = new MemoryStream();
            stream.CopyTo(ms);
            byte[] modelBytes = ms.ToArray();

            _session = new InferenceSession(modelBytes);
        }

        public float[] RunInference(byte[] imageBytes)
        {
            using var ms = new MemoryStream(imageBytes);
            using var originalImage = new Bitmap(ms);

            // Preprocessing (model requirement)
            using var processedImage = PadImageIfNecessary(originalImage);
            float[] imageData = NormalizeImage(processedImage);
            var inputTensor = CreateTensorFromImage(imageData, processedImage.Width, processedImage.Height);

            string inputName = _session.InputMetadata.Keys.First();
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor<float>(inputName, inputTensor)
            };

            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = _session.Run(inputs);
            var outputTensor = results.First().AsTensor<float>();
            return outputTensor.ToArray();
        }

        private Bitmap PadImageIfNecessary(Bitmap image)
        {
            int width = image.Width;
            int height = image.Height;

            if (width >= MinSize && height >= MinSize)
            {
                return new Bitmap(image);
            }

            // Calculate new dimensions ensuring a minimum of 224.
            int paddedWidth = Math.Max(width, MinSize);
            int paddedHeight = Math.Max(height, MinSize);

            Bitmap paddedImage = new Bitmap(paddedWidth, paddedHeight);
            using (Graphics graphics = Graphics.FromImage(paddedImage))
            {
                graphics.Clear(Color.White); //white padding

                int offsetX = (paddedWidth - width) / 2;
                int offsetY = (paddedHeight - height) / 2;
                graphics.DrawImage(image, offsetX, offsetY, width, height);
            }

            return paddedImage;
        }

        private float[] NormalizeImage(Bitmap image)
        {
            int width = image.Width;
            int height = image.Height;
            // Output array with three channels (R, G, B) in CHW order.
            float[] data = new float[3 * width * height];

            // Normalization parameters for ImageNet.
            float[] mean = { 0.485f, 0.456f, 0.406f };
            float[] std = { 0.229f, 0.224f, 0.225f };

            // Lock the bitmap data for efficient access.
            BitmapData bitmapData = image.LockBits(new Rectangle(0, 0, width, height),
                ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);

            unsafe
            {
                for (int y = 0; y < height; y++)
                {
                    byte* row = (byte*)bitmapData.Scan0 + (y * bitmapData.Stride);
                    for (int x = 0; x < width; x++)
                    {
                        int pixelIndex = y * width + x;
                        // Format24bppRgb: pixel order is Blue, Green, Red.
                        float r = row[x * 3 + 2] / 255.0f;
                        float g = row[x * 3 + 1] / 255.0f;
                        float b = row[x * 3 + 0] / 255.0f;
                        // Store in CHW order.
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
            var tensor = new DenseTensor<float>(new[] { 1, 3, height, width });
            imageData.CopyTo(tensor.Buffer.Span);
            return tensor;
        }
    }
}
