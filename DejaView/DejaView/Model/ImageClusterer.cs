using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DejaView.Model
{
    internal class ImageClusterer
    {
        public static async Task<List<List<string>>> ClusterSimilarImagesAsync(
            Dictionary<string, float[]> imageVectors,
            float similarityThreshold,
            IProgress<double> progress = null,
            CancellationToken cancellationToken = default)
        {
            // Do not use the UI Thread
            return await Task.Run(() =>
            {
                List<string> keys = imageVectors.Keys.ToList();
                int n = keys.Count;
                ConcurrentBag<(int, int)> similarPairs = new ConcurrentBag<(int, int)>();
                
                // Counter for completed outer iterations
                int completed = 0;
                
                // Compare all unique pairs of images in parallel
                Parallel.For(0, n, i =>
                {
                    for (int j = i + 1; j < n; j++)
                    {
                        cancellationToken.ThrowIfCancellationRequested();
                        float similarity = CosineSimilarity(imageVectors[keys[i]], imageVectors[keys[j]]);
                        if (similarity >= similarityThreshold)
                        {
                            similarPairs.Add((i, j));
                        }
                    }

                    // Update progress after processing each outer iteration (each image)
                    if (progress != null)
                    {
                        int done = Interlocked.Increment(ref completed);
                        double percent = (double)done / n * 100;
                        progress.Report(percent);
                    }
                });

                // Initialize Union-Find structure for clustering
                int[] clusterId = new int[n];
                for (int i = 0; i < n; i++)
                {
                    clusterId[i] = i;
                }

                // Local functions for Union-Find
                int Find(int i)
                {
                    if (clusterId[i] != i)
                        clusterId[i] = Find(clusterId[i]);
                    return clusterId[i];
                }

                void Union(int i, int j)
                {
                    int root1 = Find(i);
                    int root2 = Find(j);
                    if (root1 != root2)
                    {
                        clusterId[root2] = root1;
                    }
                }

                // Merge clusters based on similar pairs
                foreach (var (i, j) in similarPairs)
                {
                    Union(i, j);
                }

                // Aggregate images into clusters
                Dictionary<int, List<string>> clusters = new Dictionary<int, List<string>>();
                for (int i = 0; i < n; i++)
                {
                    int root = Find(i);
                    if (!clusters.ContainsKey(root))
                    {
                        clusters[root] = new List<string>();
                    }
                    clusters[root].Add(keys[i]);
                }

                return clusters.Values.ToList();
            }, cancellationToken);
        }

        private static float CosineSimilarity(float[] vectorA, float[] vectorB)
        {
            if (vectorA.Length != vectorB.Length)
                throw new ArgumentException("Vectors must be of same length");

            float dot = 0f;
            float normA = 0f;
            float normB = 0f;

            for (int i = 0; i < vectorA.Length; i++)
            {
                dot += vectorA[i] * vectorB[i];
                normA += vectorA[i] * vectorA[i];
                normB += vectorB[i] * vectorB[i];
            }

            if (normA == 0 || normB == 0)
                return 0;

            return dot / ((float)Math.Sqrt(normA) * (float)Math.Sqrt(normB));
        }
    }
}
