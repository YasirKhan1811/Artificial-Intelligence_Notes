# Clustering
## KMeans Clustering

```python
# Import KMeans
from sklearn.cluster import KMeans

# Create a KMeans instance with n clusters: model
model = KMeans(n_clusters=n)

# train the model with the data
model.fit(data)

# Determine the cluster labels of new_samples: labels
labels = model.predict(new_samples)

# Print cluster labels of new_samples
print(labels)
```

To visualize the clusters made from the data by the KMeans Algorithm

```python
# import pyplot
import matplotlib.pyplot as plt

# Assign the columns of new_samples: x and y
x = new_samples[:, 0]
y = new_samples[:, 1]

# Make a scatter plot of x and y, using labels to define the colors
plt.scatter(x, y, c=labels, alpha=0.5)

# Assign the cluster centers: centroids
centroids = model.cluster_centers_

# Assign the columns of centroids: centroids_x, centroids_y
centroids_x = centroids[:, 0]
centroids_y = centroids[:, 1]

# Make a scatter plot of centroids_x and centroids_y
plt.scatter(centroids_x, centroids_y, marker='D', s=50)
plt.show()
```

### Evaluating a Clustering (Cross Tabulation and Inertia)
To evaluate the quality of our clusters, we have a method called cross_tabulation. However, this works when we already have the original labels. We can do this, by forming a Pandas df
of two columns (i.e., cluster labels and the original categories). For example:

```python
import pandas as pd
df = pd.DataFrame({'labels': labels, 'categories': original_categories})
print(df)
```

```python
ct = pd.crosstab(df['labels'], df['categories'])
print(ct)
```

In contrast, most datasets don't come with original categories, and they are unlabeled. To evaluate the clustering quality in such a case, there is a method called **Inertia**.









