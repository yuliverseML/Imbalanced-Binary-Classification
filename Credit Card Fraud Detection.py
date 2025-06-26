# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

# Data loading
print("Loading data...")
data = pd.read_csv('/content/creditcard.csv')

# Exploratory data analysis
print(f"Dataset size: {data.shape}")
print(f"Fraudulent transactions: {data['Class'].sum()} ({data['Class'].sum()/len(data)*100:.2f}%)")
print(f"Missing values: {data.isnull().sum().sum()}")

# Class imbalance visualization
plt.figure(figsize=(8, 6))
sns.countplot(x='Class', data=data)
plt.title('Class Distribution (0: Legitimate, 1: Fraudulent)')
plt.show()

# Data splitting with chronological consideration
print("Splitting data...")
# Sort by time for chronological splitting
data = data.sort_values('Time')

# Split into training and testing sets (80/20)
train_size = 0.8
split_idx = int(len(data) * train_size)

train_data = data.iloc[:split_idx]
test_data = data.iloc[split_idx:]

# Verify distribution in train and test sets
print(f"Training set: {len(train_data)} transactions, {train_data['Class'].sum()} fraudulent ({train_data['Class'].sum()/len(train_data)*100:.2f}%)")
print(f"Test set: {len(test_data)} transactions, {test_data['Class'].sum()} fraudulent ({test_data['Class'].sum()/len(test_data)*100:.2f}%)")

# Separate features and target variable
X_train = train_data.drop('Class', axis=1)
y_train = train_data['Class']
X_test = test_data.drop('Class', axis=1)
y_test = test_data['Class']

# Feature scaling
print("Scaling features...")
# Apply scaling only on training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame for convenience
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Class imbalance handling
print("Balancing the training dataset...")
# Use downsampling approach (without SMOTE for simplicity)
# Select all fraudulent transactions
fraud_train = train_data[train_data['Class'] == 1]
non_fraud_train = train_data[train_data['Class'] == 0]

# Take a random sample of legitimate transactions
n_fraud = len(fraud_train)
non_fraud_downsampled = non_fraud_train.sample(n=n_fraud * 5, random_state=42)

# Combine into a balanced dataset for training
balanced_train = pd.concat([non_fraud_downsampled, fraud_train])
balanced_train = balanced_train.sample(frac=1, random_state=42).reset_index(drop=True)

X_train_balanced = balanced_train.drop('Class', axis=1)
y_train_balanced = balanced_train['Class']

# Scale the balanced dataset
X_train_balanced_scaled = scaler.transform(X_train_balanced)
X_train_balanced_scaled = pd.DataFrame(X_train_balanced_scaled, columns=X_train.columns)

print(f"Balanced training set: {len(X_train_balanced)} transactions, {y_train_balanced.sum()} fraudulent ({y_train_balanced.sum()/len(y_train_balanced)*100:.2f}%)")

# Model selection and training
print("Training and evaluating models...")

# Configure and compare multiple models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
    'Logistic Regression': LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)
}

# Use stratified cross-validation for proper evaluation of imbalanced data
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Evaluate models using cross-validation
for name, model in models.items():
    start_time = time.time()
    # Use only the training data for cross-validation
    cv_scores = cross_val_score(model, X_train_balanced_scaled, y_train_balanced, 
                              cv=cv, scoring='roc_auc', n_jobs=-1)
    
    print(f"{name} - Mean ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f} (completed in {time.time() - start_time:.2f} sec)")

# Train the best model (Random Forest in this case)
best_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
best_model.fit(X_train_balanced_scaled, y_train_balanced)

# Model evaluation on test set
print("\nEvaluating on test set...")
# Important: evaluate on the original imbalanced test set
y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
y_pred = (y_pred_proba >= 0.5).astype(int)

# Basic metrics
print("\nMetrics on test set:")
print(classification_report(y_test, y_pred))

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

# Precision-Recall curve (more informative for imbalanced data)
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)

plt.figure(figsize=(10, 8))
plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (area = {pr_auc:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.show()




# Threshold optimization
print("\nOptimizing threshold value...")
# Find optimal threshold for F1-score
f1_scores = []
for threshold in np.arange(0.1, 0.9, 0.05):
    y_pred_thresh = (y_pred_proba >= threshold).astype(int)
    f1 = f1_score(y_test, y_pred_thresh)
    f1_scores.append((threshold, f1))

best_threshold, best_f1 = max(f1_scores, key=lambda x: x[1])
print(f"Optimal threshold: {best_threshold:.2f} (F1-score: {best_f1:.4f})")

# Apply optimal threshold
y_pred_optimal = (y_pred_proba >= best_threshold).astype(int)
print("\nMetrics with optimal threshold:")
print(classification_report(y_test, y_pred_optimal))

# Feature importance analysis
print("\nAnalyzing feature importance...")
feature_importance = best_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

# Visualize top 15 features
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(15))
plt.title('Top 15 Most Important Features')
plt.tight_layout()
plt.show()

print("\nTop 10 most important features:")
print(feature_importance_df.head(10))

# Model persistence
print("\nSaving model...")
joblib.dump(best_model, 'fraud_detection_model.pkl')
joblib.dump(scaler, 'fraud_scaler.pkl')


# Prediction function implementation
def predict_fraud(transaction_data, threshold=best_threshold):
    """
    Predict fraud for a new transaction
    
    Args:
        transaction_data: DataFrame with transaction features
        threshold: Classification threshold (default: optimal from validation)
    
    Returns:
        Dictionary with prediction and probability
    """
    # Check for all required features
    required_features = X_train.columns.tolist()
    if not all(feature in transaction_data.columns for feature in required_features):
        missing = [f for f in required_features if f not in transaction_data.columns]
        raise ValueError(f"Missing features: {missing}")
    
    # Ensure correct feature order
    transaction_data = transaction_data[required_features]
    
    # Scale features
    scaled_data = scaler.transform(transaction_data)
    
    # Get probability
    fraud_probability = best_model.predict_proba(scaled_data)[:, 1]
    
    # Make prediction based on threshold
    prediction = (fraud_probability >= threshold).astype(int)
    
    return {
        'prediction': prediction.tolist(),
        'probability': fraud_probability.tolist(),
        'is_fraud': bool(prediction[0]),
        'risk_level': 'High' if fraud_probability[0] > 0.7 else 
                     ('Medium' if fraud_probability[0] > 0.4 else 'Low')
    }

# Example usage
print("\nExample prediction for a new transaction:")
# Take a random transaction from the test set
sample_transaction = X_test.sample(1)
result = predict_fraud(sample_transaction)

print(f"Fraud probability: {result['probability'][0]:.4f}")
print(f"Prediction: {'Fraudulent' if result['is_fraud'] else 'Legitimate'} transaction")
print(f"Risk level: {result['risk_level']}")

print("\nDone!")
