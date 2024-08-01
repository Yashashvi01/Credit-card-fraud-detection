# Credit Card Fraud Detection

This project aims to detect fraudulent credit card transactions using machine learning techniques. The dataset used contains transactions made by European cardholders in September 2013.

## Project Overview

The objective of this project is to build a machine learning model that can accurately identify fraudulent credit card transactions. The dataset is highly imbalanced, with the majority of transactions being non-fraudulent. Therefore, techniques such as SMOTE (Synthetic Minority Over-sampling Technique) and downsampling are used to handle class imbalance.

## Dataset

The dataset used in this project can be found on [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud). It contains credit card transactions over a two-day period in September 2013 by European cardholders.

- **Number of transactions**: 284,807
- **Number of fraudulent transactions**: 492 (0.172% of all transactions)

The dataset includes the following features:
- **Time**: The seconds elapsed between this transaction and the first transaction in the dataset
- **V1 to V28**: Principal components obtained with PCA
- **Amount**: Transaction amount
- **Class**: 1 for fraudulent transactions, 0 otherwise

## Dependencies

- pandas
- numpy
- scikit-learn
- imbalanced-learn

You can install the dependencies using pip:

```bash
pip install pandas numpy scikit-learn imbalanced-learn
```

## File Descriptions
credit_card_fraud_detection.ipynb : Jupyter notebook containing the entire analysis and model building process.
data/: Directory containing the dataset.

## Data Preprocessing
Detailed steps on how the data was preprocessed:

## Loading the Data:

The dataset was loaded using pandas.
## Exploratory Data Analysis (EDA):

Initial exploration of the dataset to understand the distribution of features and the class imbalance.
Visualization of the data to identify any anomalies or patterns.
## Handling Class Imbalance:

Two approaches were used: SMOTE oversampling and downsampling.
## Feature Scaling:

StandardScaler was used to scale the features to have zero mean and unit variance.
## Model Building
Description of the models built and the techniques used:

## Approach 1: SMOTE Oversampling
## SMOTE:

Synthetic Minority Over-sampling Technique was used to generate synthetic samples for the minority class (fraudulent transactions) to balance the dataset.
Model Selection:

Several machine learning algorithms were tested, including Logistic Regression, Decision Trees, Random Forests, and Gradient Boosting.
Training the Model:

The dataset was split into training and testing sets.
Models were trained using the training set and evaluated on the testing set.
## Approach 2: Downsampling
## Downsampling:

The majority class (non-fraudulent transactions) was downsampled to match the number of minority class samples to balance the dataset.
Model Selection:

Similar to the SMOTE approach, multiple machine learning algorithms were tested.
Training the Model:

The dataset was split into training and testing sets.
Models were trained using the training set and evaluated on the testing set.
Evaluation
Detailed explanation of the evaluation metrics and the results obtained:

## Confusion Matrix:

A confusion matrix was used to evaluate the performance of the models.
Accuracy, Precision, Recall, and F1-Score:

## Key Takeaways
## Effectiveness of SMOTE:

SMOTE was effective in handling class imbalance by generating synthetic samples for the minority class, which improved model performance in identifying fraudulent transactions.
## Downsampling Insights:

Downsampling the majority class to balance the dataset also yielded good results but sometimes at the cost of losing important information from the majority class.
## Model Performance:

Ensemble methods like Random Forest and Gradient Boosting generally performed better compared to Logistic Regression and Decision Trees.
The choice of handling class imbalance significantly affects model performance, with SMOTE generally providing better results compared to downsampling.
## Importance of Evaluation Metrics:

Relying solely on accuracy can be misleading in imbalanced datasets. Precision, Recall, and F1-Score provide a more holistic view of model performance.
