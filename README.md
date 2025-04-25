# Diabetes_Classification
Diabetes Prediction Model
Overview
This project focuses on predicting whether a person has diabetes based on various medical features. The dataset contains features like age, gender, smoking history, and other health-related parameters. The objective of this project is to build machine learning models to predict the likelihood of a person having diabetes.

# Dataset

The dataset includes the following features:

Gender: Gender of the person (Male/Female)

Age: Age of the person

Hypertension: Whether the person has hypertension (Yes/No)

Heart Disease: Whether the person has heart disease (Yes/No)

Smoking History: Smoking history of the person (Yes/No)

BMI: Body Mass Index

HbA1c Level: Hemoglobin A1c level

Blood Glucose Level: Blood glucose level

Diabetes: Target variable indicating if the person has diabetes (1 for Yes, 0 for No)

Data Preprocessing
Handling Missing Values:

Missing values in the dataset are replaced with the mean for continuous variables like BMI, blood glucose level, and HbA1c level.

For categorical variables, any missing entries can be handled by replacing them with a mode or by using other techniques like imputation.

Encoding Categorical Variables:

Categorical variables like Gender and Smoking History are encoded into numerical format using LabelEncoder to convert them into a format suitable for machine learning models.

Feature Scaling:

StandardScaler is applied to scale the feature set to ensure that all features are on the same scale for better model performance.

# Data Splitting:


The dataset is split into training and testing sets (80% for training and 20% for testing) using train_test_split from scikit-learn.

# Models Used

The following models are used to predict diabetes:

Logistic Regression

Decision Tree Classifier

K-Nearest Neighbors (KNN)

Naive Bayes

Linear Discriminant Analysis (LDA)

Each of these models is trained on the preprocessed dataset and evaluated based on accuracy.

# Model Evaluation

The models are evaluated using the following metrics:


Accuracy: The proportion of correct predictions made by the model.

Confusion Matrix: A table used to describe the performance of a classification model, showing the true positive, true negative, false positive, and false negative counts.

# Accuracy Comparison


The following table compares the accuracy of all five models:


Model	Accuracy

Logistic Regression	0.95870

Decision Tree Classifier	0.95200

K-Nearest Neighbors (KNN)	0.96135

Naive Bayes	0.90475

Linear Discriminant Analysis (LDA)	0.95555
