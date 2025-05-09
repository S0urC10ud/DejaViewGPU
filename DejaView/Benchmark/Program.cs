using BenchmarkDotNet.Running;
using BenchmarkDotNet.Attributes;
using DejaView.Model;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Jobs;

namespace MyApp.Benchmark
{


    [MemoryDiagnoser]
    [Config(typeof(SingleRunConfig))]
    public class ProcessAllFilesBenchmark
    {
        private IEnumerable<string> filePaths;

        [GlobalSetup]
        public void Setup()
        {
            var sampleDir = @"C:\Users\marti\Pictures\DatasetDejaView";
            filePaths = Directory.GetFiles(sampleDir);
        }

        [Benchmark]
        public Task ProcessAllFilesLongRunningForEachAsync()
        {
            return ImageFileScanner.ProcessAllFilesLongRunningForEachAsync(filePaths, null);
        }

        [Benchmark]
        public Task ProcessAllFilesNoLongRunningForEachAsync()
        {
            return ImageFileScanner.ProcessAllFilesNoLongRunningForEachAsync(filePaths, null);
        }

        [Benchmark]
        public Task ProcessAllFilesLongRunningForEach()
        {
            return ImageFileScanner.ProcessAllFilesLongRunningForEach(filePaths, null);
        }

        [Benchmark]
        public Task ProcessAllFilesNoLongRunningForEach()
        {
            return ImageFileScanner.ProcessAllFilesNoLongRunningForEach(filePaths, null);
        }

        [Benchmark]
        public Task ProcessAllFilesNoParallelization()
        {
            return ImageFileScanner.ProcessAllFilesNoParallelization(filePaths, null);
        }

        private class SingleRunConfig : ManualConfig
        {
            public SingleRunConfig()
            {
                AddJob(Job.Default
                    .WithWarmupCount(1)
                    .WithIterationCount(10) //measure 10 times each
                    .WithInvocationCount(1) // each measurement calls the method once
                    .WithUnrollFactor(1)); //compiler-level optimization, no need for slow method, dont optimize multiple calls into a loop
            }
        }
    }

    public class Program
    {
        public static void Main(string[] args)
        {
            BenchmarkRunner.Run<ProcessAllFilesBenchmark>();
        }
    }
}