import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# 1. Load the Dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

# The dataset doesn't have a header, so we define column names based on UCI documentation
column_names = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]

# Note: Missing values are represented by '?' in this dataset
df = pd.read_csv(url, names=column_names, na_values="?")

# 2. Data Preprocessing
# Fill missing values with the median (common practice for this dataset)
df = df.fillna(df.median())

# Binary Classification: The 'target' column has values 0-4. 
# 0 = No Disease, 1-4 = Presence of Heart Disease.
# We map 1-4 to 1 to make it a binary problem.
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

# 3. Split into Features and Target
X = df.drop('target', axis=1)
y = df['target']

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Feature Scaling (Crucial for Logistic Regression performance)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. Build and Train the Model
model = LogisticRegression()
model.fit(X_train, y_train)

# 7. Make Predictions and Evaluate
y_pred = model.predict(X_test)



# Get predicted probabilities for the positive class (Heart Disease = 1)
y_probs = model.predict_proba(X_test)[:, 1]

# Calculate AUC
auc_score = roc_auc_score(y_test, y_probs)
print(f"ROC-AUC Score: {auc_score:.4f}")

# Generate ROC curve values
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Diagonal line
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()

print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.2f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# 1. Define the record
new_record = pd.DataFrame([[60, 1, 3, 130, 406, 0, 0, 132, 1, 2.4, 1, 0, 3]], columns=X.columns)

# 2. IMPORTANT: Scale the record using the same scaler used during training
new_record_scaled = scaler.transform(new_record)

# 3. Predict using the scaled data
prediction = model.predict(new_record_scaled)
probability = model.predict_proba(new_record_scaled)

print(f"Prediction: {'Heart Disease' if prediction[0] == 1 else 'No Heart Disease'}")
print(f"Confidence: {np.max(probability) * 100:.2f}%")
