# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Set a visual style for seaborn
sns.set_style('whitegrid')

# Load the data
file_path = '/content/creditcard.csv'  # Update with the correct file path if needed
data = pd.read_csv(file_path)

# Display basic data info and statistics
print("Dataset Info:")
print(data.info())

print("\nDataset Description:")
print(data.describe())


# Class distribution
print("\nClass Distribution:")
print(data['Class'].value_counts())

# Visualize class distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=data)
plt.title('Class Distribution')
plt.show()

# Visualize transaction amount distribution for fraudulent vs non-fraudulent transactions
plt.figure(figsize=(12, 6))
sns.histplot(data[data['Class'] == 0]['Amount'], bins=50, color='blue', label='Non-Fraud', kde=True)
sns.histplot(data[data['Class'] == 1]['Amount'], bins=50, color='red', label='Fraud', kde=True)
plt.legend()
plt.title('Transaction Amount Distribution by Class')
plt.xlabel('Transaction Amount')
plt.ylabel('Frequency')
plt.show()

# Visualize time distribution for fraudulent vs non-fraudulent transactions
plt.figure(figsize=(12, 6))
sns.histplot(data[data['Class'] == 0]['Time'], bins=50, color='blue', label='Non-Fraud', kde=True)
sns.histplot(data[data['Class'] == 1]['Time'], bins=50, color='red', label='Fraud', kde=True)
plt.legend()
plt.title('Transaction Time Distribution by Class')
plt.xlabel('Transaction Time')
plt.ylabel('Frequency')
plt.show()


# Standardize 'Amount' and 'Time'
scaler = StandardScaler()
data[['Amount', 'Time']] = scaler.fit_transform(data[['Amount', 'Time']])

# Separate features and target variable
X = data.drop(columns=['Class'])
y = data['Class']

# Handling class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Check the new class distribution after SMOTE
print("\nClass Distribution After SMOTE:")
print(pd.Series(y_resampled).value_counts())

# Visualize new class distribution after SMOTE
plt.figure(figsize=(6, 4))
sns.countplot(x=y_resampled)
plt.title('Class Distribution After SMOTE')
plt.show()

# Train-Test Split (70% training and 30% testing, stratified by class)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42, stratify=y_resampled)


# Initialize models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'XGBoost': XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42),
    'Naive Bayes': GaussianNB()
}

# Train and evaluate each model
for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate the model
    print(f"\n{model_name} - Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Non-Fraud', 'Fraud']))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # ROC-AUC Score
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"{model_name} - ROC-AUC Score: {roc_auc:.4f}")

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} - ROC Curve')
    plt.legend()
    plt.show()



# Summary of model performance
model_performance = {}
for model_name, model in models.items():
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    model_performance[model_name] = roc_auc

# Display model performance
performance_df = pd.DataFrame.from_dict(model_performance, orient='index', columns=['ROC-AUC Score'])
performance_df = performance_df.sort_values(by='ROC-AUC Score', ascending=False)
print("\nModel Performance Summary (by ROC-AUC Score):")
print(performance_df)

# Plot performance summary
plt.figure(figsize=(10, 6))
sns.barplot(x=performance_df.index, y=performance_df['ROC-AUC Score'])
plt.title('Model Performance Comparison (ROC-AUC Score)')
plt.ylabel('ROC-AUC Score')
plt.xticks(rotation=45)
plt.show()



