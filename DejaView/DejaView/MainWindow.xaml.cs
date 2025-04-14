using System;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using DejaView.Model;

namespace DejaView
{
    public partial class MainWindow : Window, INotifyPropertyChanged
    {
        public event PropertyChangedEventHandler? PropertyChanged;

        private float _similarity = 0.95f;
        public float Similarity {
            get => _similarity;
            set
            {
                _similarity = value;
                OnPropertyChanged();
            }
        }
        private int _progress = 0;
        public int Progress
        {
            get => _progress;
            set
            {
                _progress = value;
                OnPropertyChanged();
                OnPropertyChanged(nameof(ProgressPercent)); // Notify that ProgressPercent has changed too
            }
        }
        public string ProgressPercent => Progress + "%";

        public IProgress<int> ProgressReporter { get; }

        private string _progressTextStep = "Step 1/4";
        public string ProgressTextStep
        {
            get => _progressTextStep;
            set
            {
                _progressTextStep = value;
                OnPropertyChanged();
            }
        }
        private string _imagesFoundText = "1234 images found, could not read 12 images";
        public string ImagesFoundText
        {
            get => _imagesFoundText;
            set
            {
                _imagesFoundText = value;
                OnPropertyChanged();
            }
        }


        private string _selectedDirectory = string.Empty;
        private CancellationTokenSource _cancellationTokenSource = new CancellationTokenSource();

        public MainWindow()
        {
            InitializeComponent();
            DataContext = this;
            ProgressReporter = new Progress<int>(value => Progress = value);
        }

        protected virtual void OnPropertyChanged([CallerMemberName] string? propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        private void BtnSelectDirectory_Click(object sender, RoutedEventArgs e)
        {
            using (var dialog = new FolderBrowserDialog())
            {
                dialog.Description = "Select a directory containing image files";
                dialog.ShowNewFolderButton = false;
                if (dialog.ShowDialog() == System.Windows.Forms.DialogResult.OK)
                {
                    _selectedDirectory = dialog.SelectedPath;
                    txtSelectedDirectory.Text = _selectedDirectory;
                    // Enable the start processing button once a directory is selected.
                    btnStartProcessing.IsEnabled = true;
                }
            }
        }

        private void BtnStartProcessing_Click(object sender, RoutedEventArgs e)
        {
            if ((string)btnStartProcessing.Content == "Start Processing")
            {
                btnStartProcessing.Content = "Cancel";
                ProcessFiles();
            }
            else
            {
                // TODO: cancel processing
                _cancellationTokenSource.Cancel();
                ProgressReporter.Report(0);
                btnStartProcessing.Content = "Start Processing";
            }

        }

        private async void ProcessFiles()
        {
            if (string.IsNullOrWhiteSpace(_selectedDirectory))
            {
                System.Windows.MessageBox.Show("Please select a directory first.", "Directory Not Selected", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            try
            {
                ProgressTextStep = "Step 1/2";
                // TODO: maybe Progress reporter??
                // Do not come back to the UI thread -> ConfigureAwait(false)
                RetrievedImagePathsResult imageFiles = await ImageFileScanner.GetAllImageFilesAsync(_selectedDirectory, _cancellationTokenSource.Token).ConfigureAwait(false);

                // TODO: warning if no images were found

                // Get all files first to make proper progress bars
                // Do not come back to the UI thread -> ConfigureAwait(false)
                ProcessedImagesResult results = await ImageFileScanner.ProcessAllFilesAsync(imageFiles.files, ProgressReporter, _cancellationTokenSource.Token).ConfigureAwait(false);

                ProgressReporter.Report(0);
                ProgressTextStep = "Step 2/2";

                // Come back to the UI thread for displaying the results
                List<List<string>> clusters = await ImageClusterer.ClusterSimilarImagesAsync(results.embeddings, Similarity, ProgressReporter, _cancellationTokenSource.Token); //TODO: add cancelation token
            }
            catch (OperationCanceledException)
            {
                _cancellationTokenSource.Dispose();
                _cancellationTokenSource = new CancellationTokenSource();
            }
            catch (Exception ex)
            {
                System.Windows.MessageBox.Show($"Error processing images: {ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        private void BtnPrevious_Click(object sender, RoutedEventArgs e)
        {
            Similarity = 0.9f;
            Progress = 60;
            ProgressTextStep = "Step 2/4";
            ImagesFoundText = "found more images, could read all of them";
        }
    }
}
