**Cross Validation:**
```python
from sklearn.model_selection import cross_val_score, KFold
kf = KFold(n_splits=6, shuffle=True, random_state=42)
model = LinearRegression()
cv_results = cross_val_score(model, X_train, y_train, cv=kf)
```

**Confusion Matrix**
The confusion matrix is particularly useful when dealing with binary classification problems (two classes), but it can also be extended to multi-class problems.
- Accuracy = (correct predictions) / (total observations)

```python
from sklearn.metrics import classification_report, confusion_matrix
knn = KNeighborsClassifier(n_neighbors=7)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
```

- Model Evaluation

```python
print(confusion_matrix(y_test, y_pred))

```

- Classification Report

```python  
print(classification_report(y_test, y_pred)

```

- Sensitivity: Percentage of correctly predicted actual positive class.
  $$ Sensitivity = \True Positives // True Positives + False Negatives\, dx $$

- Specificity: Percentage of correctly predicted actual negative class.
  $$ Specificity = \True Negatives // True Negatives + False Positives\, dx $$

    - Precision is different, which is the percentage of predicted positives.
    - We decide which model to choose based on the Sensitivity and Specificity scores.
    - If predicting the +ive class is our interest, then we choose the model based on the higher sensitivity score.
    - But if predicting the -ive class is our interest, then we choose the model based on the higher specificity score.

**ROC AUC Curve**
ROC score method is used to validate a binary classifier (Logistic Regression)

```python
for, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve')
plt.show()
```
