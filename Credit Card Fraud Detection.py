# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
import joblib
import numpy as np

# ==================================================
# Data Loading and Inspection:
# The dataset is loaded and inspected for missing values and structure.
# ==================================================

# Load the dataset
data = pd.read_csv('/content/creditcard.csv')

# Display the first few rows of the dataset to understand its structure
print(data.head())

# Check for missing values in the dataset
print(data.isnull().sum())

# ==================================================
# Data Balancing:
# The dataset is balanced by downsampling the majority class (non-fraudulent transactions).
# ==================================================

# Separate fraudulent and non-fraudulent transactions
fraud = data[data['Class'] == 1]
non_fraud = data[data['Class'] == 0]

# Downsample the non-fraudulent transactions to balance the dataset
non_fraud_downsampled = resample(non_fraud, replace=False, n_samples=len(fraud), random_state=42)

# Combine the downsampled non-fraudulent transactions with the fraudulent ones
balanced_data = pd.concat([non_fraud_downsampled, fraud])

# ==================================================
# Exploratory Data Analysis (EDA):
# Visualizations are created to understand the class distribution and feature correlations.
# ==================================================

# Visualize the distribution of classes in the balanced dataset
sns.countplot(x='Class', data=balanced_data)
plt.title('Class Distribution')
plt.show()

# Create a heatmap to visualize correlations between features
corr = balanced_data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr, cmap='coolwarm', annot=False)
plt.title('Correlation Heatmap')
plt.show()

# ==================================================
# Model Training:
# A RandomForestClassifier is trained on the balanced dataset.
# ==================================================

# Split the data into features (X) and target variable (y)
X = balanced_data.drop('Class', axis=1)
y = balanced_data['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForestClassifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on the training data
model.fit(X_train, y_train)

# ==================================================
# Model Evaluation:
# The model is evaluated using metrics like precision, recall, F1-score, ROC-AUC, and a confusion matrix.
# ==================================================

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model using classification report and confusion matrix
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Calculate ROC-AUC and F1-Score
roc_auc = roc_auc_score(y_test, y_pred)
print(f'ROC-AUC: {roc_auc}')

f1 = f1_score(y_test, y_pred)
print(f'F1-Score: {f1}')

# ==================================================
# Model Saving and Loading:
# The trained model is saved to a file and loaded for future use.
# ==================================================

# Save the trained model to a file
joblib.dump(model, 'fraud_detection_model.pkl')

# Load the model from the file
loaded_model = joblib.load('fraud_detection_model.pkl')

# Display the column names in the training data for reference
print(X_train.columns)

# ==================================================
# Prediction on New Data:
# The model is used to make predictions on new data points, both manually created and randomly generated.
# ==================================================

# Create a sample new data point with the correct column names and order
new_data = pd.DataFrame({
    'Time': [100000],  # Time
    'V1': [1.23], 
    'V2': [-0.45], 
    'V3': [0.78], 
    'V4': [-1.56], 
    'V5': [0.34], 
    'V6': [-0.67], 
    'V7': [0.12], 
    'V8': [-0.89], 
    'V9': [1.45], 
    'V10': [-0.23], 
    'V11': [0.56], 
    'V12': [-1.23], 
    'V13': [0.67], 
    'V14': [-0.78], 
    'V15': [0.89], 
    'V16': [-0.12], 
    'V17': [1.34], 
    'V18': [-0.45], 
    'V19': [0.56], 
    'V20': [-0.67], 
    'V21': [0.78], 
    'V22': [-0.89], 
    'V23': [0.12], 
    'V24': [-0.34], 
    'V25': [0.45], 
    'V26': [-0.56], 
    'V27': [0.67], 
    'V28': [-0.78], 
    'Amount': [100.0]  # Amount
})

# Ensure the column order matches the training data
new_data = new_data[X_train.columns]

# Make a prediction on the new data
prediction = loaded_model.predict(new_data)
print('Fraudulent Transaction' if prediction[0] == 1 else 'Legitimate Transaction')

# Generate random data for all features
new_data = pd.DataFrame(np.random.randn(1, 30), columns=X_train.columns)

# Make a prediction on the random data
prediction = loaded_model.predict(new_data)
print('Fraudulent Transaction' if prediction[0] == 1 else 'Legitimate Transaction')
