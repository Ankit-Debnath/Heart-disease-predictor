# ❤️ Heart Disease Prediction Using Machine Learning

# 📚 Importing Required Libraries

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns

# 📂 Loading the Dataset


df = pd.read_csv("heart.csv")

df.head(10)

# 🔍 Dataset Information



df.info()

print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\nMissing Values:\n")
print(df.isnull().sum())

print("\nDuplicate Values Before Removal:", df.duplicated().sum())

# Remove duplicate rows
df = df.drop_duplicates()

print("\nDuplicate Values After Removal:", df.duplicated().sum())

print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

# 🎯 Target Variable Distribution



df['target'].value_counts()

df['target'].value_counts().plot(kind='bar')

plt.title("Target Distribution")
plt.xlabel("Target")
plt.ylabel("Count")

plt.show()


# 📈 Thal Feature Distribution


df['thal'].value_counts()

df['thal'].value_counts().plot(kind='bar')

plt.title("Thal Distribution")
plt.xlabel("Thal")
plt.ylabel("Count")

plt.show()

# 📦 Outlier Detection Using Boxplots

cols = ['chol', 'trestbps', 'thalach', 'oldpeak']

plt.figure(figsize=(10, 6))

df[cols].boxplot()

plt.title("Boxplots for Outlier Detection")
plt.ylabel("Values")

plt.show()

# ✂️ Splitting Dataset

x = df.drop('target', axis=1)
y = df['target']


x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# 🤖 Logistic Regression Model

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(
    random_state=42,
    max_iter=5000))])

# 🎛️ Hyperparameter Tuning using GridSearchCV



from sklearn.model_selection import GridSearchCV

param_grid = [
    {
        'model__solver': ['liblinear'],
        'model__penalty': ['l1', 'l2'],
        'model__C': [0.01, 0.1, 1, 10]
    },
    {
        'model__solver': ['lbfgs'],
        'model__penalty': ['l2'],
        'model__C': [0.01, 0.1, 1, 10]
    }
]
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
model = GridSearchCV(
    pipeline,
    param_grid,
    cv=cv,
    scoring='recall',
    n_jobs=-1
)

model.fit(x_train, y_train)

# ✅ Best Parameters Found

print("Best Parameters:")
print(model.best_params_)

best_idx = model.best_index_

print(f"Mean CV Accuracy: {model.cv_results_['mean_test_score'][best_idx]:.4f}")
print(f"Std CV Accuracy : {model.cv_results_['std_test_score'][best_idx]:.4f}")

# ***📌 Logistic Regression Predictions***

y_pred = model.predict(x_test)
y_prob = model.predict_proba(x_test)[:, 1]

# 📊 Model Evaluation Metrics

lr_accuracy = accuracy_score(y_test, y_pred)
lr_precision = precision_score(y_test, y_pred)
lr_recall = recall_score(y_test, y_pred)
lr_f1 = f1_score(y_test, y_pred)
lr_auc = roc_auc_score(y_test, y_prob)

print("Accuracy :", lr_accuracy)
print("Precision:", lr_precision)
print("Recall   :", lr_recall)
print("F1 Score :", lr_f1)
print("ROC AUC  :", lr_auc)

print(classification_report(y_test, y_pred))

# 🔲 Confusion Matrix

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# 📉 ROC Curve

fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure(figsize=(6,5))

plt.plot(fpr, tpr, label=f"AUC = {lr_auc:.2f}")
plt.plot([0,1],[0,1],'--')

plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.legend()

plt.show()

# ⚡ Support Vector Machine (SVM) Model

# Pipeline
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', SVC(
    probability=True,
    random_state=42))])

# Parameters
svm_param_grid = {
    'model__C': [0.01,0.1,1,10,100],
    'model__kernel': ['linear', 'rbf'],
    'model__gamma': ['scale', 0.001,0.01,0.1,1]
}



# GridSearchCV
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
svm_grid = GridSearchCV(
    estimator=svm_pipeline,
    param_grid=svm_param_grid,
    cv=cv,
    scoring='recall',
    n_jobs=-1
)

# Train
svm_grid.fit(x_train, y_train)


print("Best Parameters:")
print(svm_grid.best_params_)

best_idx = svm_grid.best_index_

print(f"Mean CV Accuracy: {svm_grid.cv_results_['mean_test_score'][best_idx]:.4f}")
print(f"Std CV Accuracy : {svm_grid.cv_results_['std_test_score'][best_idx]:.4f}")

# 📊 SVM Performance Evaluation

svm_pred = svm_grid.predict(x_test)
svm_prob = svm_grid.predict_proba(x_test)[:, 1]

svm_accuracy = accuracy_score(y_test, svm_pred)
svm_precision = precision_score(y_test, svm_pred)
svm_recall = recall_score(y_test, svm_pred)
svm_f1 = f1_score(y_test, svm_pred)
svm_auc = roc_auc_score(y_test, svm_prob)

print("Accuracy :", svm_accuracy)
print("Precision:", svm_precision)
print("Recall   :", svm_recall)
print("F1 Score :", svm_f1)
print("ROC AUC  :", svm_auc)

print("\nClassification Report:\n")
print(classification_report(y_test, svm_pred))

# 🔲 SVM Confusion Matrix

cm_svm = confusion_matrix(y_test, svm_pred)

plt.figure(figsize=(5,4))

sns.heatmap(
    cm_svm,
    annot=True,
    fmt='d',
    cmap='Greens'
)

plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

## ***📉 ROC Curve***

fpr_svm, tpr_svm, _ = roc_curve(y_test, svm_prob)

plt.figure(figsize=(6,5))

plt.plot(fpr_svm, tpr_svm, label=f"AUC = {svm_auc:.2f}")
plt.plot([0,1], [0,1], '--')

plt.title("SVM ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.legend()

plt.show()

# 🌲 Random Forest Model


# Pipeline
rf_pipeline = Pipeline([
    ('model', RandomForestClassifier(random_state=42))
])

# Parameters
rf_param_grid = {
    "model__n_estimators":[100,200,300],
    "model__max_depth":[5,10,None],
    "model__min_samples_split":[2,5],
    "model__min_samples_leaf":[1,2],
    'model__criterion': ['gini', 'entropy']
}

# GridSearchCV
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
rf_grid = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=rf_param_grid,
    cv=cv,
    scoring='recall',
    n_jobs=-1
)

# Train
rf_grid.fit(x_train, y_train)


print("Best Parameters:")
print(rf_grid.best_params_)

best_idx = rf_grid.best_index_

print(f"Mean CV Accuracy: {rf_grid.cv_results_['mean_test_score'][best_idx]:.4f}")
print(f"Std CV Accuracy : {rf_grid.cv_results_['std_test_score'][best_idx]:.4f}")

# 📊 Random Forest Performance Evaluation

# Predictions
rf_pred = rf_grid.predict(x_test)
rf_prob = rf_grid.predict_proba(x_test)[:, 1]

# Metrics
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)
rf_auc = roc_auc_score(y_test, rf_prob)

print("Accuracy :", rf_accuracy)
print("Precision:", rf_precision)
print("Recall   :", rf_recall)
print("F1 Score :", rf_f1)
print("ROC AUC  :", rf_auc)

print("\nClassification Report:\n")
print(classification_report(y_test, rf_pred))

# 🔲 Random Forest Confusion Matrix

cm_rf = confusion_matrix(y_test, rf_pred)

plt.figure(figsize=(5,4))

sns.heatmap(
    cm_rf,
    annot=True,
    fmt='d',
    cmap='Oranges'
)

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# ***📉 ROC Curve***

fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_prob)

plt.figure(figsize=(6,5))

plt.plot(fpr_rf, tpr_rf, label=f"AUC = {rf_auc:.2f}")
plt.plot([0,1], [0,1], '--')

plt.title("Random Forest ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.legend()

plt.show()

# ⭐ Feature Importance Analysis

# Get best Random Forest model
best_rf_model = rf_grid.best_estimator_

# Extract feature importance
importance = best_rf_model.named_steps['model'].feature_importances_

# Create DataFrame
feature_importance = pd.DataFrame({
    'Feature': x.columns,
    'Importance': importance
})

# Sort values
feature_importance = feature_importance.sort_values(
    by='Importance',
    ascending=False
)

feature_importance

# 📈 Feature Importance Visualization

plt.figure(figsize=(10,6))

sns.barplot(
    x='Importance',
    y='Feature',
    data=feature_importance
)

plt.title("Feature Importance - Random Forest")

plt.show()

# ⚖️ Model Performance Comparison

results = pd.DataFrame({

    'Model': [
    'Logistic Regression',
    'SVM',
    'Random Forest'
],

    'Accuracy': [
    lr_accuracy,
    svm_accuracy,
    rf_accuracy
],

    'Precision': [
    lr_precision,
    svm_precision,
    rf_precision
],

'Recall': [
    lr_recall,
    svm_recall,
    rf_recall
],

'F1 Score': [
    lr_f1,
    svm_f1,
    rf_f1
],

'ROC AUC': [
    lr_auc,
    svm_auc,
    rf_auc
]
})

results.round(3)

# 📊 Comparison Graph of Machine Learning Models

results.set_index('Model').plot(
    kind='bar',
    figsize=(10,6)
)

plt.title("ML Model Comparison")
plt.ylabel("Score")
plt.ylim(0,1)

plt.show()

# Logistic Regression
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob)

# SVM
fpr_svm, tpr_svm, _ = roc_curve(y_test, svm_prob)

# Random Forest
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_prob)

plt.figure(figsize=(8,6))

plt.plot(
    fpr_lr,
    tpr_lr,
    label=f'Logistic Regression AUC = {lr_auc:.2f}'
)

plt.plot(
    fpr_svm,
    tpr_svm,
    label=f'SVM AUC = {svm_auc:.2f}'
)

plt.plot(
    fpr_rf,
    tpr_rf,
    label=f'Random Forest AUC = {rf_auc:.2f}'
)

plt.plot([0,1], [0,1], '--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve Comparison")

plt.legend()

plt.show()

# 🔳 Combined Confusion Matrix Comparison

fig, axes = plt.subplots(1, 3, figsize=(15,4))

# Logistic Regression
sns.heatmap(
    confusion_matrix(y_test, y_pred),
    annot=True,
    fmt='d',
    cmap='Blues',
    ax=axes[0]
)

axes[0].set_title("Logistic Regression")

# SVM
sns.heatmap(
    confusion_matrix(y_test, svm_pred),
    annot=True,
    fmt='d',
    cmap='Greens',
    ax=axes[1]
)

axes[1].set_title("SVM")


# Random Forest
sns.heatmap(
    confusion_matrix(y_test, rf_pred),
    annot=True,
    fmt='d',
    cmap='Oranges',
    ax=axes[2]
)

axes[2].set_title("Random Forest")

plt.tight_layout()

plt.show()

# 📋 Confusion Matrix Comparison Table

cm_lr = confusion_matrix(y_test, y_pred)
cm_svm = confusion_matrix(y_test, svm_pred)
cm_rf = confusion_matrix(y_test, rf_pred)

cm_comparison = pd.DataFrame({
    'Model': ['Logistic Regression', 'SVM', 'Random Forest'],
    'TN': [cm_lr[0,0], cm_svm[0,0], cm_rf[0,0]],
    'FP': [cm_lr[0,1], cm_svm[0,1], cm_rf[0,1]],
    'FN': [cm_lr[1,0], cm_svm[1,0], cm_rf[1,0]],
    'TP': [cm_lr[1,1], cm_svm[1,1], cm_rf[1,1]]
})

print("\nConfusion Matrix Comparison:")
print(cm_comparison)

# 📊 Confusion Matrix Values Comparison

cm_comparison.set_index('Model').plot(
    kind='bar',
    figsize=(10,6)
)

plt.title("Confusion Matrix Comparison")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.show()

cv_results = pd.DataFrame({
    'Model': ['Logistic Regression', 'SVM', 'Random Forest'],
    'Mean CV Score': [
        model.cv_results_['mean_test_score'][model.best_index_],
        svm_grid.cv_results_['mean_test_score'][svm_grid.best_index_],
        rf_grid.cv_results_['mean_test_score'][rf_grid.best_index_]
    ],
    'Std CV Score': [
        model.cv_results_['std_test_score'][model.best_index_],
        svm_grid.cv_results_['std_test_score'][svm_grid.best_index_],
        rf_grid.cv_results_['std_test_score'][rf_grid.best_index_]
    ]
})

print("\nCross Validation Comparison")
print(cv_results.round(4))

import joblib

joblib.dump(model.best_estimator_, "heart_disease_model.pkl")
print("Model saved successfully!")