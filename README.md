# heart-disease-logistic-regression
A machine learning pipeline using Logistic Regression to predict heart disease risk from clinical patient data, featuring ROC-AUC evaluation and feature scaling.

This project implements a Logistic Regression machine learning model to predict the presence of heart disease in patients based on clinical data. Using the UCI Cleveland Heart Disease Dataset, the model analyzes 13 medical features—such as age, cholesterol levels, chest pain type, and maximum heart rate—to determine if a patient is likely to have heart disease.

The core of this project is to demonstrate a full machine learning pipeline: from data cleaning and handling missing values to feature scaling and rigorous evaluation using ROC-AUC and Confusion Matrices.

📊 Summary of Implementation
Data Source: UCI Machine Learning Repository (Cleveland Dataset).

Task: Binary Classification (Target 0: Healthy | Target 1: Heart Disease).

Preprocessing: * Handled missing values using median imputation.

Applied StandardScaler to normalize feature ranges.

Split data into 80% Training and 20% Testing sets.

Model: Logistic Regression.

Key Metrics:

Accuracy: Overall correctness of the model.

Recall (Sensitivity): Crucial for healthcare to ensure we minimize "False Negatives."

ROC-AUC: Measures the model's ability to distinguish between classes across thresholds.
