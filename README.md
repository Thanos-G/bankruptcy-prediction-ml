# Bankruptcy Prediction using Machine Learning

This project implements several machine learning models to predict the bankruptcy of Greek companies based on financial indicators.

## Problem

Bankruptcy prediction is a binary classification problem where companies are classified as:

- Healthy
- Bankrupt

## Dataset

The dataset contains financial indicators describing the performance of Greek companies.

Target variable:

0 → Healthy  
1 → Bankrupt

## Data Preprocessing

Steps performed:

- Removed "Year" column
- Encoded target variable
- Feature scaling with StandardScaler
- Train/Test split (75/25)
- Class balancing using downsampling (3:1 ratio)

## Models Implemented

The following classifiers were trained:

- Linear Discriminant Analysis
- Logistic Regression
- K-Nearest Neighbors (k = 5, 7, 9)
- Naive Bayes
- Support Vector Machine
- Neural Network (MLP)

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## Output

The evaluation results are exported to:
