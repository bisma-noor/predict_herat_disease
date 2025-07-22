import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("Heart Disease Prediction Model")
print("="*50)

np.random.seed(42)
n_samples = 1000

data = {
    'age': np.random.randint(30, 80, n_samples),
    'sex': np.random.choice([0, 1], n_samples),
    'chest_pain_type': np.random.choice([0, 1, 2, 3], n_samples),
    'resting_bp': np.random.randint(90, 200, n_samples),
    'cholesterol': np.random.randint(120, 400, n_samples),
    'fasting_blood_sugar': np.random.choice([0, 1], n_samples),
    'rest_ecg': np.random.choice([0, 1, 2], n_samples),
    'max_heart_rate': np.random.randint(80, 200, n_samples),
    'exercise_angina': np.random.choice([0, 1], n_samples),
    'oldpeak': np.random.uniform(0, 6, n_samples),
    'slope': np.random.choice([0, 1, 2], n_samples),
    'ca': np.random.choice([0, 1, 2, 3], n_samples),
    'thal': np.random.choice([0, 1, 2, 3], n_samples),
}

target = []
for i in range(n_samples):
    risk_score = (
        data['age'][i] * 0.02 +
        data['sex'][i] * 0.3 +
        data['chest_pain_type'][i] * 0.2 +
        (data['resting_bp'][i] > 140) * 0.4 +
        (data['cholesterol'][i] > 240) * 0.3 +
        data['exercise_angina'][i] * 0.5 +
        data['oldpeak'][i] * 0.2 +
        np.random.normal(0, 0.3)
    )
    target.append(1 if risk_score > 2.0 else 0)

data['target'] = target
df = pd.DataFrame(data)

print(f"Dataset loaded successfully with shape: {df.shape}")

df = df.drop_duplicates()

print("\nPerforming Exploratory Data Analysis...")

plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='target')
plt.title("Distribution of Target Variable")
plt.xlabel("Heart Disease (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='age', hue='target', multiple='stack', bins=20)
plt.title("Age Distribution by Heart Disease")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='chest_pain_type', hue='target')
plt.title("Chest Pain Type vs Target")
plt.xlabel("Chest Pain Type")
plt.ylabel("Count")
plt.show()

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(random_state=42)
}

results = []

for name, model in models.items():
    if name == 'SVM':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })

results_df = pd.DataFrame(results)
print("\nModel Performance Summary:")
print(results_df.round(4).to_string(index=False))

results_df.set_index('Model')[['Accuracy', 'Precision', 'Recall', 'F1-Score']].plot(kind='bar', figsize=(10, 6))
plt.title("Model Comparison on Evaluation Metrics")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

best_model_idx = results_df['F1-Score'].idxmax()
best_model_name = results_df.iloc[best_model_idx]['Model']
best_model = models[best_model_name]

if best_model_name == 'SVM':
    best_model.fit(X_train_scaled, y_train)
    y_pred_best = best_model.predict(X_test_scaled)
else:
    best_model.fit(X_train, y_train)
    y_pred_best = best_model.predict(X_test)

print(f"\nClassification Report for Best Model ({best_model_name}):")
print(classification_report(y_test, y_pred_best))

conf_mat = confusion_matrix(y_test, y_pred_best)
sns.heatmap(conf_mat, annot=True, cmap='Blues', fmt='d')
plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
