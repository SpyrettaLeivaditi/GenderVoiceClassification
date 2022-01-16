## Classifying Speaker Gender with Voice Features, using Logistic Regression

### Description 
The code in this repository deals with the problem of predicting a speaker's gender, based only on the characteristics of their voice. We formulate the problem as a classification task between two classes (Male and Female), and we use Logistic Regression to perform the classification.

This task is part of an assignment in a Machine Learning (ML) course, and focuses on implementing a classification algorithm for a particular problem, evaluating its performance, and comparing it with any existing implementations.

### Data 

#### Data Location
The dataset we used in this implementation is freely and openly available in [Kaggle](https://www.kaggle.com/primaryobjects/voicegender). It consists of 3,168 recorded voice samples, collected from male and female speakers, and pre-processed to extract a number of acoustic features.

#### Data columns - Data types

The dataset consists of 20 independent variables:

- **meanfreq**: mean frequency (in kHz); float
- **sd**: standard deviation of frequency; float
- **median**: median frequency (in kHz); float
- **Q25**: first quantile (in kHz); float
- **Q75**: third quantile (in kHz); float
- **IQR**: interquantile range (in kHz); float
- **skew**: skewness; float
- **kurt**: kurtosis; float
- **sp.ent**: spectral entropy; float
- **sfm**: spectral flatness; float
- **mode**: mode frequency; float
- **centroid**: frequency centroid; float
- **meanfun**: mean fundamental frequency measured across acoustic signal; float
- **minfun**: minimum fundamental frequency measured across acoustic signal; float
- **maxfun**: maximum fundamental frequency measured across acoustic signal; float
- **meandom**: mean of dominant frequency measured across acoustic signal; float
- **mindom**: minimum of dominant frequency measured across acoustic signal; float
- **maxdom**: maximum of dominant frequency measured across acoustic signal; float
- **dfrange**: range of dominant frequency measured across acoustic signal; float
- **modindx**: modulation index; float

1 target variable:
- labels: male/female (string)

### Running the code

To run the code first install the required dependencies:

```
pip install -r requirements.txt
```
Then you can run the classification task by running:

```
python main.py
```

The output consists of a set of calculated evaluation metrics printed in the console, and a number of plots, all illustrating the classification's performance.

### What our code does

In our code, we execute the following workflow:

- First we read the data from a csv file and split them into features and labels.
- Then we plot a data correlation matrix, in the form of a heatmap, that shows the pairwise correlation between the different columns of the data.
- Then, we encode the string labels (male and female) into numbers (0 for female, 1 for male). 
- Then, we normalize the feature values to the range [0,1]  using a MinMax scaler. This is necessary as the different features have different ranges that may negatively affect  the classifier.
- Then, we split the data into train and test sets, with the train set consisting of 80% of the data and the test one of 20%. This splitting is random in every run.
- Then, we initialize a Logistic Regression model.
- Then, we define the scoring metrics that we want to evaluate the model with. These are accuracy, precision, recall, F1, and confusion matrix.
- Then we run and evaluate the classification model by performing a 10-fold cross-validation. We use cross-validation to ensure our evaluation is not biased from the random splitting of the data. 
- Finally, we print the average values of the scoring metrics and plot the confusion matrix for each fold.

### Comparison with other implementation

Our implementation can be compared with the implementation reported [here](https://www.kaggle.com/monukhan/gender-voice-classification). The main differences between them are the following:

- That implementation uses Support Vector Machines (SVM) as the classification algorithm and achieves an accuracy of 98.26%, while our implementation uses Logistic Regression and achieves an accuracy of 96.92%.
- That implementation reports only the accuracy score and a single confusion matrix, while our implementation reports also precision, recall and F1, and plots the confusion matrix per fold.
- That implementation standardizes the features by removing the mean and scaling to unit variance, while our implementation scales the features to the range [0,1].
- That implementation uses GridSearch to find the best parameters for the SVM algorithm, while our implementation does not do any parameter tuning.






