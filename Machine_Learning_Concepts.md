# Machine Learning

12/16/2023

## Classification Problems
- Model is built by training it on labeled data, where the pair of features and target variables fit the model.
- The model is used to predict the labels by giving it the unseen features/data.
- The model is evaluated using the actual observations and the predicted observations.

**k-NearestNeighors (KNN)**

```python
from sklearn.neighbors import KNeighborsClassifier
X = df[["feature1", "feature2", "feature3"]].values
y = df["target"].values
print(X.shape, y.shape) # check if the sizes of the arrays are equal

# splitting training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

# instantiate the model
knn = KNeighborsClassifier(n_neighbors=k)

# model training
knn.fit(X_train, y_train)

# predictions on the test set
predictions = knn.predict(X_test)

# model evaluation
print("Accuracy:", accuracy_score(y_test, predictions))
```

**Cross Validation:**
```python
from sklearn.model_selection import cross_val_score, KFold
kf = KFold(n_splits=6, shuffle=True, random_state=42)
model = LinearRegression()
cv_results = cross_val_score(model, X_train, y_train, cv=kf)
```

**Confusion Matrix**
The confusion matrix is particularly useful when dealing with binary classification problems (two classes), but it can also be extended to multi-class problems.
- Accuracy = (correct predictions) / (all predictions)

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




















