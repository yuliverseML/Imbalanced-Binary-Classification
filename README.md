Credit Card Fraud Detection
# Credit Card Fraud Detection
This project focuses on detecting fraudulent credit card transactions using machine learning. The dataset used is the [Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) from Kaggle. The goal is to build a model that can accurately classify transactions as fraudulent or legitimate.

## Models Implemented
The following machine learning models were implemented and evaluated:
- **Random Forest Classifier**: Used as the primary model due to its robustness and ability to handle imbalanced datasets.
- (Optional: Add other models like Logistic Regression, XGBoost, etc., if implemented.)

## Features
### Data Exploration
- **Dataset Overview**: The dataset contains 284,807 transactions, with 492 fraudulent cases (highly imbalanced).
- **Class Distribution**: Visualized using a count plot to understand the imbalance between fraudulent and legitimate transactions.
- **Correlation Analysis**: A heatmap was created to explore correlations between features.

### Data Preprocessing
- **Balancing the Dataset**: The dataset was balanced by downsampling the majority class (legitimate transactions) to match the number of fraudulent transactions.
- **Feature Engineering**: New features like `Log_Amount`, `Hour`, and `Amount_Time_Ratio` were created to improve model performance.
- **Train-Test Split**: The data was split into training (80%) and testing (20%) sets.

### Model Training
- **Random Forest**: Trained on the balanced dataset with default hyperparameters.
- **Hyperparameter Tuning**: (Optional: Add details if GridSearchCV or RandomizedSearchCV was used.)

### Model Evaluation
- **Metrics Used**: Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix.
- **Results**:
  - Precision: 0.96 (Fraudulent), 0.89 (Legitimate)
  - Recall: 0.88 (Fraudulent), 0.96 (Legitimate)
  - F1-Score: 0.92 (Fraudulent), 0.92 (Legitimate)
  - ROC-AUC: 0.92

### Visualization
- **Class Distribution**: A count plot was used to visualize the balanced dataset.
- **Correlation Heatmap**: A heatmap was created to visualize feature correlations.
- **Confusion Matrix**: Used to evaluate the model's performance on the test set.

## Results

### Model Comparison
- **Random Forest**: Achieved an F1-Score of 0.92 and ROC-AUC of 0.92.
- (Optional: Add comparisons with other models if implemented.)

### Best Model
- **Random Forest**: Selected as the best-performing model due to its high F1-Score and ROC-AUC.

### Feature Importance
- **Top Features**: (Optional: Add a list of the most important features as determined by the Random Forest model.)

## Outcome

### Best Performing Model
- **Random Forest**: Achieved an accuracy of 92% on the test set, with a balanced precision and recall for both classes.

## Future Work
- **Hyperparameter Tuning**: Further optimize the Random Forest model using techniques like Bayesian Optimization.
- **Alternative Models**: Experiment with other algorithms like XGBoost, LightGBM, or Neural Networks.
- **Real-Time Detection**: Implement the model in a real-time fraud detection system.
- **Feature Engineering**: Explore additional features or transformations to improve model performance.

## Notes
- The dataset is highly imbalanced, so balancing techniques like downsampling were applied.
- Feature engineering played a key role in improving model performance.
- The Random Forest model was chosen for its balance between accuracy and interpretability.

## Contributing
Contributions are welcome! If you'd like to contribute, please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Submit a pull request with a detailed description of your changes.
## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

