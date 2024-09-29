# Brainwave_matrix_Intern_task2

# Credit Card Fraud Detection with Machine Learning

This repository contains a project to detect fraudulent transactions in a credit card dataset using various machine learning models. The dataset is highly imbalanced, so techniques like SMOTE (Synthetic Minority Oversampling Technique) are used to handle the class imbalance.

## Table of Contents

- [Overview](#overview)
- [Data](#data)
- [Installation](#installation)
- [Data Preprocessing](#data-preprocessing)
- [Model Training & Evaluation](#model-training--evaluation)
- [Results](#results)
- [License](#license)

## Overview

Credit card fraud detection is a significant challenge due to the large number of non-fraudulent transactions compared to fraudulent ones. This project demonstrates how to preprocess the data, handle class imbalance, and evaluate various machine learning models to detect fraudulent transactions effectively.

## Data

The dataset used is `creditcard.csv`, which contains credit card transactions made by European cardholders over two days in September 2013. This dataset is highly imbalanced, where fraudulent transactions are much fewer than non-fraudulent ones.

- **Time**: Number of seconds elapsed between this transaction and the first transaction in the dataset.
- **Amount**: Transaction amount.
- **Class**: The target variable (0 for non-fraud, 1 for fraud).

## Installation

To run this project, you need to have the following dependencies installed:

```bash
pip install pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost
```

## Data Preprocessing 

- **Standardization**: The Amount and Time features are standardized using StandardScaler.
- **SMOTE**: Class imbalance is addressed using SMOTE, which oversamples the minority class (fraudulent transactions).
- **Train-Test Split**: The data is split into 70% training and 30% testing sets, stratified by the class distribution.

## Model Training & Evaluation

Seven machine learning models were trained and evaluated:

- Logistic Regression
- Random Forest
- Support Vector Machine
- K-Nearest Neighbors
- Gradient Boosting
- XGBoost
- Naive Bayes

Each model is evaluated using the following metrics:

- **Classification Report**: Precision, Recall, and F1-Score.
- **Confusion Matrix**: Visual representation of model performance.
- **ROC-AUC Score**: Area Under the ROC Curve for measuring model performance.
- **ROC Curve**: Graph showing the tradeoff between True Positive Rate and False Positive Rate.

Results
The performance of the models is summarized by their ROC-AUC scores:

| Model                    | ROC-AUC Score |
|---------------------------|---------------|
| Random Forest              | 1.000000        |
| XGBoost              | 0.999995       |
| K-Nearest Neighbors         | 0.999654        |
| Gradient Boosting           | 0.998278        |
| Logistic Regression     | 0.950459      |
| Naive Bayes                | 0.950459        |

A bar chart comparing the ROC-AUC scores of the models is displayed to highlight the best-performing model.

## Output

![image](https://github.com/user-attachments/assets/f97a186d-1cc6-4a4a-8564-552ef2199b6b)

