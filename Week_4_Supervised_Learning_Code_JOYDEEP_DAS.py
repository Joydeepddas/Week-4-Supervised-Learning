# Week 4 - Supervised Learning Model Implementation
# Breast Cancer Classification using Logistic Regression
# Prepared by: JOYDEEP DAS

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report,
    RocCurveDisplay)

data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target
print('Dataset shape:', X.shape)
print('Target names:', data.target_names)
print('Class distribution:\n', y.value_counts().sort_index())
print('Missing values:', X.isnull().sum().sum())

plt.figure(figsize=(7,4)); y.value_counts().sort_index().plot(kind='bar')
plt.title('Breast Cancer Class Distribution')
plt.xlabel('Target Class (0 = Malignant, 1 = Benign)'); plt.ylabel('Number of Samples')
plt.xticks(rotation=0); plt.tight_layout(); plt.show()

plt.figure(figsize=(7,4)); plt.hist(X['mean radius'], bins=25)
plt.title('Distribution of Mean Radius'); plt.xlabel('Mean Radius'); plt.ylabel('Frequency')
plt.tight_layout(); plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42)

model = Pipeline([
    ('scaler', StandardScaler()),
    ('logistic_regression', LogisticRegression(max_iter=5000, random_state=42))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = cross_validate(model, X_train, y_train, cv=cv, scoring={
    'accuracy':'accuracy', 'precision':'precision', 'recall':'recall',
    'f1':'f1', 'roc_auc':'roc_auc'})

print('\n5-Fold Cross-Validation Results')
for metric in ['accuracy','precision','recall','f1','roc_auc']:
    scores = cv_results[f'test_{metric}']
    print(f'{metric.upper():10s}: {scores.mean():.4f} +/- {scores.std():.4f}')

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print('\nFinal Test-Set Performance')
print(f'Accuracy : {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall   : {recall:.4f}')
print(f'F1 Score : {f1:.4f}')
print(f'ROC-AUC  : {roc_auc:.4f}')
print('\nClassification Report:')
print(classification_report(y_test, y_pred, target_names=data.target_names))

cm = confusion_matrix(y_test, y_pred)
print('\nConfusion Matrix:\n', cm)
plt.figure(figsize=(6,5)); plt.imshow(cm)
plt.title('Confusion Matrix'); plt.xlabel('Predicted Label'); plt.ylabel('True Label')
plt.xticks([0,1], ['Malignant','Benign']); plt.yticks([0,1], ['Malignant','Benign'])
for i in range(2):
    for j in range(2): plt.text(j, i, cm[i,j], ha='center', va='center')
plt.colorbar(); plt.tight_layout(); plt.show()

RocCurveDisplay.from_predictions(y_test, y_prob)
plt.title('ROC Curve - Logistic Regression'); plt.tight_layout(); plt.show()

logistic_model = model.named_steps['logistic_regression']
coefficients = pd.Series(logistic_model.coef_[0], index=X.columns).sort_values()
print('\nMost negative coefficients:\n', coefficients.head(10))
print('\nMost positive coefficients:\n', coefficients.tail(10))
plt.figure(figsize=(9,7)); coefficients.plot(kind='barh')
plt.title('Logistic Regression Feature Coefficients')
plt.xlabel('Coefficient'); plt.ylabel('Feature'); plt.tight_layout(); plt.show()

dataset_copy = X.copy(); dataset_copy['target'] = y
dataset_copy.to_csv('Breast_Cancer_Week4.csv', index=False)
print('\nDataset saved as: Breast_Cancer_Week4.csv')
print('Week 4 supervised learning project completed successfully.')
