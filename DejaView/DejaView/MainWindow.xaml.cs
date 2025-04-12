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

        private double _similarity = 0.95;
        public double Similarity {
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
        public MainWindow()
        {
            InitializeComponent();
            this.DataContext = this;
        }

        protected virtual void OnPropertyChanged([CallerMemberName] string? propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        private void btnSelectDirectory_Click(object sender, RoutedEventArgs e)
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

        private async void btnStartProcessing_Click(object sender, RoutedEventArgs e)
        {
            if (string.IsNullOrWhiteSpace(_selectedDirectory))
            {
                System.Windows.MessageBox.Show("Please select a directory first.", "Directory Not Selected", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            // Disable the start button to prevent multiple processing requests.
            btnStartProcessing.IsEnabled = false;
            try
            {
                // TODO: Progress reporter and cancellation token
                // Do not come back to the UI thread -> ConfigureAwait(false)
                var imageFiles = await ImageFileScanner.GetAllImageFilesAsync(_selectedDirectory).ConfigureAwait(false);

                // Get all files first to make proper progress bars
                // Do not come back to the UI thread -> ConfigureAwait(false)
                var results = await ImageFileScanner.ProcessAllFilesAsync(imageFiles).ConfigureAwait(false);

                // Come back to the UI thread for displaying the results
                var clusters = await ImageClusterer.ClusterSimilarImagesAsync(results, 0.99f); //TODO: supply actual similarity Threshold

                System.Windows.MessageBox.Show($"Processed {results.Count} images into {clusters.Count} clusters.", "Processing Complete", MessageBoxButton.OK, MessageBoxImage.Information);
            }
            catch (Exception ex)
            {
                System.Windows.MessageBox.Show($"Error processing images: {ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
            finally
            {
                btnStartProcessing.IsEnabled = true;
            }
        }

        private void btnPrevious_Click(object sender, RoutedEventArgs e)
        {
            Similarity = 0.9;
            Progress = 60;
        }
    }
}
