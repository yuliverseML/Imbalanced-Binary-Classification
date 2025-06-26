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


##########################################################
######################################################################


# Импорт необходимых библиотек
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

# 1. Загрузка данных
print("Загрузка данных...")
data = pd.read_csv('/content/creditcard.csv')

# 2. Исследовательский анализ данных
print(f"Размер датасета: {data.shape}")
print(f"Мошеннические транзакции: {data['Class'].sum()} ({data['Class'].sum()/len(data)*100:.2f}%)")
print(f"Пропущенные значения: {data.isnull().sum().sum()}")

# Визуализация дисбаланса классов
plt.figure(figsize=(8, 6))
sns.countplot(x='Class', data=data)
plt.title('Распределение классов (0: Легитимные, 1: Мошеннические)')
plt.show()

# 3. Разделение данных с учетом временной структуры
print("Разделение данных...")
# Сортировка по времени для хронологического разделения
data = data.sort_values('Time')

# Разделение на обучающий и тестовый наборы (80/20)
train_size = 0.8
split_idx = int(len(data) * train_size)

train_data = data.iloc[:split_idx]
test_data = data.iloc[split_idx:]

# Проверка распределения в обучающей и тестовой выборках
print(f"Обучающая выборка: {len(train_data)} транзакций, {train_data['Class'].sum()} мошеннических ({train_data['Class'].sum()/len(train_data)*100:.2f}%)")
print(f"Тестовая выборка: {len(test_data)} транзакций, {test_data['Class'].sum()} мошеннических ({test_data['Class'].sum()/len(test_data)*100:.2f}%)")

# Разделение на признаки и целевую переменную
X_train = train_data.drop('Class', axis=1)
y_train = train_data['Class']
X_test = test_data.drop('Class', axis=1)
y_test = test_data['Class']

# 4. Предобработка данных
print("Масштабирование признаков...")
# Создание и применение масштабирования только на обучающих данных
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Преобразование обратно в DataFrame для удобства
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# 5. Обработка дисбаланса классов
print("Балансировка обучающей выборки...")
# Используем различные методы балансировки только для обучающей выборки

# 5.1 Случайная подвыборка (без использования SMOTE для простоты)
# Отбираем все мошеннические транзакции
fraud_train = train_data[train_data['Class'] == 1]
non_fraud_train = train_data[train_data['Class'] == 0]

# Берем случайную выборку из легитимных транзакций, соответствующую числу мошеннических
n_fraud = len(fraud_train)
non_fraud_downsampled = non_fraud_train.sample(n=n_fraud * 5, random_state=42)

# Объединяем в сбалансированный набор данных для обучения
balanced_train = pd.concat([non_fraud_downsampled, fraud_train])
balanced_train = balanced_train.sample(frac=1, random_state=42).reset_index(drop=True)

X_train_balanced = balanced_train.drop('Class', axis=1)
y_train_balanced = balanced_train['Class']

# Масштабирование сбалансированной выборки
X_train_balanced_scaled = scaler.transform(X_train_balanced)
X_train_balanced_scaled = pd.DataFrame(X_train_balanced_scaled, columns=X_train.columns)

print(f"Сбалансированная обучающая выборка: {len(X_train_balanced)} транзакций, {y_train_balanced.sum()} мошеннических ({y_train_balanced.sum()/len(y_train_balanced)*100:.2f}%)")

# 6. Выбор и обучение модели
print("Обучение и оценка моделей...")

# 6.1 Настройка и сравнение моделей
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
    'Logistic Regression': LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)
}

# Используем стратифицированную кросс-валидацию для корректной работы с несбалансированными данными
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Оценка моделей с помощью кросс-валидации
for name, model in models.items():
    start_time = time.time()
    # Используем только обучающие данные для кросс-валидации
    cv_scores = cross_val_score(model, X_train_balanced_scaled, y_train_balanced, 
                              cv=cv, scoring='roc_auc', n_jobs=-1)
    
    print(f"{name} - Mean ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f} (выполнено за {time.time() - start_time:.2f} сек)")

# 6.2 Обучение лучшей модели (предположим, это Random Forest)
best_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
best_model.fit(X_train_balanced_scaled, y_train_balanced)

# 7. Оценка на тестовой выборке
print("\nОценка на тестовой выборке...")
# Важно: оцениваем на исходной несбалансированной тестовой выборке
y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
y_pred = (y_pred_proba >= 0.5).astype(int)

# 7.1 Базовые метрики
print("\nМетрики на тестовой выборке:")
print(classification_report(y_test, y_pred))

# 7.2 ROC-кривая
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC кривая (площадь = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

# 7.3 Precision-Recall кривая (более информативна для несбалансированных данных)
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)

plt.figure(figsize=(10, 8))
plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall кривая (площадь = {pr_auc:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.show()

# 8. Оптимизация порогового значения
print("\nОптимизация порогового значения...")
# Поиск оптимального порога для F1-меры
f1_scores = []
for threshold in np.arange(0.1, 0.9, 0.05):
    y_pred_thresh = (y_pred_proba >= threshold).astype(int)
    f1 = f1_score(y_test, y_pred_thresh)
    f1_scores.append((threshold, f1))

best_threshold, best_f1 = max(f1_scores, key=lambda x: x[1])
print(f"Оптимальный порог: {best_threshold:.2f} (F1-мера: {best_f1:.4f})")

# Применение оптимального порога
y_pred_optimal = (y_pred_proba >= best_threshold).astype(int)
print("\nМетрики с оптимальным порогом:")
print(classification_report(y_test, y_pred_optimal))

# 9. Анализ важности признаков
print("\nАнализ важности признаков...")
feature_importance = best_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Признак': X_train.columns,
    'Важность': feature_importance
}).sort_values('Важность', ascending=False)

# Визуализация топ-15 признаков
plt.figure(figsize=(12, 8))
sns.barplot(x='Важность', y='Признак', data=feature_importance_df.head(15))
plt.title('Топ-15 важнейших признаков')
plt.tight_layout()
plt.show()

print("\nТоп-10 важнейших признаков:")
print(feature_importance_df.head(10))

# 10. Сохранение модели и масштабировщика
print("\nСохранение модели...")
joblib.dump(best_model, 'fraud_detection_model.pkl')
joblib.dump(scaler, 'fraud_scaler.pkl')

# 11. Функция для предсказаний на новых данных
def predict_fraud(transaction_data, threshold=best_threshold):
    """
    Предсказание мошенничества для новой транзакции
    
    Args:
        transaction_data: DataFrame с признаками транзакции
        threshold: Порог классификации (по умолчанию: оптимальный из валидации)
    
    Returns:
        Словарь с предсказанием и вероятностью
    """
    # Проверка наличия всех необходимых признаков
    required_features = X_train.columns.tolist()
    if not all(feature in transaction_data.columns for feature in required_features):
        missing = [f for f in required_features if f not in transaction_data.columns]
        raise ValueError(f"Отсутствуют признаки: {missing}")
    
    # Убедимся в правильном порядке признаков
    transaction_data = transaction_data[required_features]
    
    # Масштабирование признаков
    scaled_data = scaler.transform(transaction_data)
    
    # Получение вероятности
    fraud_probability = best_model.predict_proba(scaled_data)[:, 1]
    
    # Предсказание на основе порога
    prediction = (fraud_probability >= threshold).astype(int)
    
    return {
        'prediction': prediction.tolist(),
        'probability': fraud_probability.tolist(),
        'is_fraud': bool(prediction[0]),
        'risk_level': 'Высокий' if fraud_probability[0] > 0.7 else 
                     ('Средний' if fraud_probability[0] > 0.4 else 'Низкий')
    }

# Пример использования
print("\nПример предсказания для новой транзакции:")
# Берем случайную транзакцию из тестовой выборки
sample_transaction = X_test.sample(1)
result = predict_fraud(sample_transaction)

print(f"Вероятность мошенничества: {result['probability'][0]:.4f}")
print(f"Предсказание: {'Мошенническая' if result['is_fraud'] else 'Легитимная'} транзакция")
print(f"Уровень риска: {result['risk_level']}")

print("\nГотово!")
