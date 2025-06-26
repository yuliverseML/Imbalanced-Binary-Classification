Credit Card Fraud Detection
# Credit Card Fraud Detection System

## Overview
This repository contains a machine learning solution for detecting fraudulent credit card transactions. The system uses supervised learning to identify potentially fraudulent activities based on anonymized transaction features. 

The dataset used is the [Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) from Kaggle. 

## Models Implemented
The following machine learning models were implemented and evaluated:
- Random Forest Classifier
- Gradient Boosting Classifier
- Logistic Regression

## Features
- Chronological data splitting to simulate real-world deployment
- Feature scaling and normalization
- Class imbalance handling
- Threshold optimization for improved precision-recall balance
- Model performance visualization
- Feature importance analysis
- Production-ready prediction function


## Data Exploration
The system works with the Credit Card Fraud Detection dataset which includes:
- 284,807 transactions, of which 492 (0.17%) are fraudulent
- 31 features: 'Time', 'Amount', and 28 anonymized features (V1-V28)
- No missing values

The dataset is highly imbalanced, with fraudulent transactions representing only 0.17% of all transactions.

## Data Preprocessing
1. **Chronological splitting**: Data is sorted by time and split into training (80%) and testing (20%) sets
2. **Feature scaling**: StandardScaler is applied to normalize all features
3. **Class balancing**: Training data is balanced using downsampling of legitimate transactions (5:1 ratio)

## Model Training
1. Models are trained on the balanced subset of the training data
2. Stratified 5-fold cross-validation is used to assess model performance
3. ROC-AUC score is used as the primary evaluation metric during training

## Model Evaluation
1. Models are evaluated on the original imbalanced test set
2. Classification metrics are calculated (precision, recall, F1-score)
3. Threshold optimization is performed to maximize F1-score
4. ROC and Precision-Recall curves are generated to visualize performance

## Visualization
The system provides several visualizations:
1. Class distribution bar chart
2. ROC curve with AUC score
3. Precision-Recall curve with AUC score
4. Feature importance bar chart for the top 15 features


## Results
### Model Comparison

| Model               | ROC-AUC (CV)           |
|---------------------|------------------------|
| Random Forest       | 0.9752 ± 0.0075        |
| Gradient Boosting   | 0.9771 ± 0.0096        |
| Logistic Regression | 0.9747 ± 0.0064        |


### Best Model Performance
The Random Forest model was selected as the final model with the following metrics:

**With default threshold (0.5):**
- Precision: 0.27
- Recall: 0.83
- F1-score: 0.41

**With optimized threshold (0.85):**
- Precision: 0.86
- Recall: 0.73
- F1-score: 0.79

### Feature Importance
Top 10 most important features:
1. V14 (0.196)
2. V10 (0.129)
3. V17 (0.124)
4. V12 (0.108)
5. V16 (0.090)
6. V11 (0.061)
7. V3 (0.054)
8. V4 (0.052)
9. V2 (0.028)
10. V18 (0.025)

## Outcome
The system successfully identifies fraudulent transactions with high precision (86%) while maintaining good recall (73%). The threshold optimization significantly improves the balance between precision and recall, resulting in a 93% increase in F1-score from 0.41 to 0.79.

## Future Work
1. **Advanced sampling techniques**: Implement SMOTE, ADASYN, or other advanced techniques for handling imbalanced data
2. **Ensemble methods**: Combine multiple models to improve overall performance
3. **Deep learning approaches**: Explore neural networks for fraud detection
4. **Feature engineering**: Create additional features based on transaction patterns
5. **Anomaly detection**: Implement unsupervised learning methods to identify unusual patterns
6. **Cost-sensitive learning**: Integrate the actual cost of false positives and false negatives
7. **Real-time monitoring**: Develop a system for detecting concept drift in fraud patterns
8. **Explainable AI**: Enhance model interpretability to provide insights into fraud detection decisions

## Notes
- The dataset features are PCA-transformed for confidentiality, making feature interpretation challenging
- The chronological splitting approach better simulates real-world deployment than random splitting
- The optimized threshold of 0.85 significantly improves model precision while maintaining acceptable recall
- Despite its simplicity, Logistic Regression shows competitive performance and is significantly faster to train

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.



