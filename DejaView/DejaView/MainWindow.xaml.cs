using DejaView.Model;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Runtime.CompilerServices;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace DejaView
{
    public partial class MainWindow : Window, INotifyPropertyChanged
    {
        public event PropertyChangedEventHandler? PropertyChanged;
        private CancellationTokenSource StartOverCancellationToken;
        private WrapPanel imageWrapPanel;

        private int currClusterId = 0;

        private List<List<string>>? clusters;

        private float _similarity = 0.95f;
        public float Similarity
        {
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
                OnPropertyChanged(nameof(ProgressPercent)); // Notify that ProgressPercent should change too
            }
        }
        public string ProgressPercent => Progress + "%";

        public IProgress<int> ProgressReporter { get; }

        private string _progressTextStep = "Step 0/2";
        public string ProgressTextStep
        {
            get => _progressTextStep;
            set
            {
                _progressTextStep = value;
                OnPropertyChanged();
            }
        }
        private string _imagesFoundText = "";
        public string ImagesFoundText
        {
            get => _imagesFoundText;
            set
            {
                _imagesFoundText = value;
                OnPropertyChanged();
            }
        }

        private Visibility _forwardBackwardVisibility = Visibility.Hidden;
        public Visibility ForwardBackwardVisibility
        {
            get => _forwardBackwardVisibility;
            set
            {
                _forwardBackwardVisibility = value;
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
            imageWrapPanel = new WrapPanel
            {
                Orientation = System.Windows.Controls.Orientation.Horizontal
            };

            ScrollViewer scrollViewer = new ScrollViewer
            {
                HorizontalScrollBarVisibility = ScrollBarVisibility.Auto,
                VerticalScrollBarVisibility = ScrollBarVisibility.Disabled,
            };

            scrollViewer.Content = imageWrapPanel;
            imageContainer.Children.Add(scrollViewer);
            StartOverCancellationToken = new CancellationTokenSource();
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
                    // Enable the start processing button once a directory is selected
                    btnStartProcessing.IsEnabled = true;
                }
            }
        }

        private void BtnStartProcessing_Click(object sender, RoutedEventArgs e)
        {
            ClearImages();
            StartOverCancellationToken.Cancel();

            ForwardBackwardVisibility = Visibility.Hidden;
            if ((string)btnStartProcessing.Content == "Start Processing")
            {
                sliderSimilarity.IsEnabled = false;
                btnStartProcessing.Content = "Cancel";
                ProcessFiles();
            }
            else
            {
                _cancellationTokenSource.Cancel();
                ProgressReporter.Report(0);
                btnStartProcessing.Content = "Start Processing";
                ProgressTextStep = "Step 0/2";
                ImagesFoundText = "";
                sliderSimilarity.IsEnabled = true;
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
                ImagesFoundText = "";
                ProgressTextStep = "Step 1/2";
                // We have to come back to the UI thread -> no ConfigureAwait(false) needed
                RetrievedImagePathsResult retrievedImagePathsResult = await ImageFileScanner.GetAllImagePathsAsync(_selectedDirectory, _cancellationTokenSource.Token);

                ImagesFoundText = retrievedImagePathsResult.files.Count() == 1 ? "Found 1 image." : $"Found {retrievedImagePathsResult.files.Count()} images.";
                if (retrievedImagePathsResult.nSkippedDirectories > 0)
                    ImagesFoundText += retrievedImagePathsResult.nSkippedDirectories == 1 ? " Could not access 1 directory." : $" Could not access {retrievedImagePathsResult.nSkippedDirectories} directories.";
                if (retrievedImagePathsResult.nIOExceptions > 0)
                    ImagesFoundText += retrievedImagePathsResult.nIOExceptions == 1 ? " Encountered 1 IO exception." : $" Encountered {retrievedImagePathsResult.nIOExceptions} IO exceptions.";


                if (retrievedImagePathsResult.files.Count == 0)
                {
                    System.Windows.MessageBox.Show("Please choose a directory containing jpg, jpeg or png files.", "No images found", MessageBoxButton.OK, MessageBoxImage.Information);
                    btnStartProcessing.Content = "Start Processing";
                    sliderSimilarity.IsEnabled = true;
                    return;
                }

                // Get all files first to enable proper progress reporting
                ProcessedImagesResult processedImagesResult = await ImageFileScanner.ProcessAllFilesLongRunningForEachAsync(retrievedImagePathsResult.files, ProgressReporter, _cancellationTokenSource.Token);

                if (processedImagesResult.nSkippedImages > 0)
                    ImagesFoundText += processedImagesResult.nSkippedImages == 1 ? "\nCould not process 1 image. " : $"\nCould not process {processedImagesResult.nSkippedImages} images. ";
                else
                    ImagesFoundText += "\n";

                ProgressReporter.Report(0);
                ProgressTextStep = "Step 2/2";

                clusters = await ImageClusterer.ClusterSimilarImagesAsync(processedImagesResult.embeddings, Similarity, ProgressReporter, _cancellationTokenSource.Token);
                if (clusters.Count > 0)
                {
                    ImagesFoundText += clusters.Count() == 1 ? "Grouped images into 1 cluster." : $"Grouped images into {clusters.Count()} clusters.";
                    currClusterId = 0;
                    DisplayCluster();
                }
                else
                    System.Windows.MessageBox.Show($"No clusters found - consider lowering the similarity threshold or adding more similar images.", "No groups found", MessageBoxButton.OK, MessageBoxImage.Information);
            }
            catch (OperationCanceledException)
            {
                _cancellationTokenSource.Dispose();
                _cancellationTokenSource = new CancellationTokenSource();
                ProgressReporter.Report(0);
            }
            catch (Exception ex)
            {
                System.Windows.MessageBox.Show($"Error processing images: {ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
            btnStartProcessing.Content = "Start Processing";
            sliderSimilarity.IsEnabled = true;
        }

        private async void DisplayCluster()
        {
            if (clusters is null)
                return;

            ForwardBackwardVisibility = Visibility.Visible;
            StartOverCancellationToken = new CancellationTokenSource();
            ClearImages();
            spinnerOverlay.Visibility = Visibility.Visible;

            btnPrevious.IsEnabled = currClusterId != 0;
            btnNext.IsEnabled = currClusterId != clusters.Count - 1;

            try
            {
                foreach (string imagePath in clusters[currClusterId])
                {
                    StartOverCancellationToken.Token.ThrowIfCancellationRequested();
                    await Task.Delay(10); // Free up the UI thread periodically
                    imageWrapPanel.Children.Add(CreateImageContainer(imagePath));
                }
            }
            catch (OperationCanceledException)
            {
                ClearImages();
            }

            spinnerOverlay.Visibility = Visibility.Collapsed;
        }

        private UIElement CreateImageContainer(string imagePath)
        {
            StackPanel container = new StackPanel
            {
                Orientation = System.Windows.Controls.Orientation.Vertical,
                HorizontalAlignment = System.Windows.HorizontalAlignment.Center,
                Margin = new Thickness(5)
            };

            Border imageBorder = new Border
            {
                Width = 150,
                Height = 150,
                ClipToBounds = true  // Elements will not leave the border
            };

            System.Windows.Controls.Image img = new System.Windows.Controls.Image
            {
                Stretch = Stretch.UniformToFill,
                HorizontalAlignment = System.Windows.HorizontalAlignment.Center,
                VerticalAlignment = VerticalAlignment.Center
            };

            try
            {
                BitmapImage bitmap = new BitmapImage();
                bitmap.BeginInit();
                bitmap.CacheOption = BitmapCacheOption.OnLoad;
                bitmap.UriSource = new Uri(imagePath, UriKind.RelativeOrAbsolute);
                bitmap.EndInit();
                img.Source = bitmap;
            }
            catch (Exception ex)
            {
                System.Windows.MessageBox.Show("Error loading image: " + ex.Message);
            }

            imageBorder.Child = img;
            container.Children.Add(imageBorder);

            StackPanel buttonPanel = new StackPanel
            {
                Orientation = System.Windows.Controls.Orientation.Horizontal,
                HorizontalAlignment = System.Windows.HorizontalAlignment.Center
            };

            System.Windows.Controls.Button btnOpen = new System.Windows.Controls.Button()
            {
                Content = "Open",
                Width = 45,
                Height = 20,
                FontSize = 12,
                Margin = new Thickness(0, 2, 5, 0),
                Tag = imagePath
            };
            btnOpen.Click += BtnOpen_Click;

            System.Windows.Controls.Button btnDelete = new System.Windows.Controls.Button
            {
                Content = "Delete",
                Width = 45,
                Height = 20,
                FontSize = 12,
                Margin = new Thickness(5, 2, 0, 0),
                Tag = new Tuple<string, StackPanel>(imagePath, container)
            };
            btnDelete.Click += BtnDelete_Click;

            buttonPanel.Children.Add(btnOpen);
            buttonPanel.Children.Add(btnDelete);
            container.Children.Add(buttonPanel);

            return container;
        }

        private void ClearImages()
        {
            imageWrapPanel.Children.Clear();
        }

        private void BtnPrevious_Click(object sender, RoutedEventArgs e)
        {
            currClusterId--;
            DisplayCluster();
        }
        private void BtnNext_Click(object sender, RoutedEventArgs e)
        {
            currClusterId++;
            DisplayCluster();
        }

        private void BtnOpen_Click(object sender, RoutedEventArgs e)
        {
            if (sender is System.Windows.Controls.Button btn && btn.Tag is string imagePath)
            {
                try
                {
                    Process.Start(new ProcessStartInfo(imagePath) { UseShellExecute = true });
                }
                catch (Exception ex)
                {
                    System.Windows.MessageBox.Show("Error opening image: " + ex.Message);
                }
            }
        }


        private void BtnDelete_Click(object sender, RoutedEventArgs e)
        {
            if (clusters == null)
                return;

            if (sender is System.Windows.Controls.Button btn && btn.Tag is Tuple<string, StackPanel> tagData)
            {
                MessageBoxResult result = System.Windows.MessageBox.Show(
                    "Are you sure you want to delete this file?",
                    "Confirm Delete",
                    MessageBoxButton.YesNo,
                    MessageBoxImage.Warning);

                string imagePath = tagData.Item1;
                StackPanel container = tagData.Item2;
                if (result == MessageBoxResult.Yes)
                {

                    try
                    {
                        File.Delete(imagePath);
                        if (container.Children.Count > 0 && container.Children[0] is Border imageBorder)
                        {
                            imageBorder.Opacity = 0.4;
                            container.Children.RemoveAt(1); // Remove StackPanel with Buttons
                        }
                        clusters[currClusterId].Remove(imagePath);
                    }
                    catch (Exception ex)
                    {
                        System.Windows.MessageBox.Show($"Could not delete the file: {ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);

                    }
                }
            }
        }
    }
}
