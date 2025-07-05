https://testingmobilenetv2.streamlit.app/
### c_sql_d.py — Database Setup
This script:

Reads image metadata (name, path, label) from zipped datasets

Inserts this metadata into a structured database table for easy querying

### vision_prep.py — Data Preprocessing and Splitting
This is the core preprocessing pipeline. It performs the following steps:

1-Dataset Loading
Iteratively reads each zipped dataset from a temporary folder(tmp)

2-Image Properties
We calculate image properties like discrete Gaussian histograms (RGB) and their moments
(mean,standard deviation,skewness,kurtosis)

3-Image Processing
Each image is processed by entering image,feature wise normalized histogram(rgb) moments,batch size.
Following enhancements in order:

Histogram Equalization:
We apply histogram equalization on the Y channel of the YUV color space to avoid shifting the color balance.
This process increases the standard deviation of the image histogram, enhancing local contrast and revealing more detail.
It may also reduce overfitting by introducing greater variation in texture and brightness.

Gamma Correction:
Gamma correction shifts the histogram toward brighter values, enhancing visibility in darker regions.
We apply it proportionally to highlight important details without distorting the overall image tone.

Median Blur:
Median blurring removes small, isolated noise — pixels that differ sharply from their neighbors — while preserving edges.
This helps smooth out unnatural pixel variations without losing important structure.

Bilateral Blur:
Bilateral filtering blurs regions of similar color while preserving edges with sharp color differences.
This softens the image gently, making feature extraction more consistent, while keeping object boundaries intact.

These steps improve image contrast and reduce noise

4-Dataset Splitting Using Histogram Clustering
We split the data into train, traincv, cv, and test sets using a custom unsupervised strategy:

For each image batch:

Concatenate image histograms rgb channels and construct a feature wise normalized correlation matrix

Apply K-Means clustering with k=1 to find the centroid of the batch

We also perform Principal Component Analysis (PCA) on feature vectors (of samples and their centroid).
Visually inspect the batch distribution.

Feature-wise Normalization:Images are normalized using ImageNet statistics (mean and std per channel).
This is compatible with most pretrained CNN models

Images closest to the centroid are assigned to CV-Test

Remaining images go to Train-TrainCV

We then shuffle both groups and further split them:

Train-TrainCV → train, traincv

CV-Test → cv, test

This approach promotes balanced generalization by separating “average” and “diverse” samples.

5-TFRecord Serialization
Final datasets are serialized into TFRecord format inside of processed_datasets folder

These are used as input for vision_training.py

### vision_training.py

gets tf record file and classify as train,traincv,cv,test

specifies batch size 

we used mobilenetv2 for depthwise separable convolution(expanding+depth wise convolution+projecting+feed forwarded(residual))

we can configure width multiplier to change channel size of convolution layers of mobilenetv2

model is trained by using tensorflow autograd

performs tensorboard,confusion matrix and see images who are miss predicted

saves a model to model folder for test_vision_dashb.py


### test_vision_dashb.py
gets all keras model files from models folder and then we can pick model in streamlit dashboard.
Picked model can predict image who is being sent to streamlit dashboard based on its own categories
