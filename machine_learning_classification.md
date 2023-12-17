# Machine Learning (Classification)

## Overview
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

# model performance
training_accuracy = knn.score(X_train, y_train)
testing_accuracy = knn.score(X_test, y_test)

```

**Model Complexity**

Choosing the right 'k' value in KNN is crucial for achieving good model performance. It often involves experimentation
and model evaluation on a validation set to find the 'k' that provides the best trade-off between bias (underfitting) and
variance (overfitting) for your specific dataset and problem.

A larger 'k' value leads to underfitting, therefore the model is less complex.
- Choosing a larger value of 'k' means that the model considers a large number of neighbors when making predictions
- A larger 'k' can lead to underfitting because the model becomes overly simple and generalized. It may not capture the local variations in the data, and its predictions may be overly biased.
- The model's decision boundary becomes very smooth, and it may not adapt well to the underlying complexity of the data.
- Signs of underfitting include poor performance on both the training and test data, as well as a model that makes overly generalized predictions.
- To address underfitting, consider reducing 'k' (making it smaller) or using more complex machine learning models.

A smaller 'k' value leads to a model overfitting the training data.
- Conversely, choosing a smaller 'k' considers a small number of neighbors for predictions.
- A smaller 'k' leads to overfitting because the model becomes overly sensitive to the noise and fluctuations in the training data. It may start to memorize the training data instead of learning the underlying patterns.
- The model's decision boundary becomes more complex, potentially capturing noise and outliers, which can result in poor generalization to new/unseen data.
- Signs of overfitting include excellent performance on the training data but poor performance on the test data (low generalization).
- To address overfitting, consider increasing 'k' (making it larger) or using techniques like cross-validation to select an appropriate 'k' and prevent the model from becoming too complex.

```python
train_accuracies = {}
test_accuracies = {}
neighbors = np.arange(1, 26)
for neighbor in neighbors:
  knn = KNeighborsClassifier(n_neighbors=neighbor)
  knn.fit(X_train, y_train)
  train_accuracies[neighbor] = knn.score(X_train, y_train)
  test_accuracies[neighbor] = knn.score(X_test, y_test)
```

- Plotting
```python
plt.figure(figsize=(8, 6))
plt.title("KNN: Varying Number of Neighbors")
plt.plot(neighbors, train_accuracies.values(), label="Training Accuracy")
plt.plot(neighbors, test_accuracies.values(), label="Testing Accuracy")
plt.legend()
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")
plt.show()
```
