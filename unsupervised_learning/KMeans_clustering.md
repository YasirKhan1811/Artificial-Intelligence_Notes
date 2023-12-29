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

A good clustering has samples in each cluster are bunched together, they are tight. This property is called **inertia**. Inertia measures the distance of each sample
from the **centroid** (the center of the cluster). We aim for the low value of inertia, however, there is a trade-off between the number of clusters and inertia. So, we choose
an optimum value between the low and high values of inertia.

Here is the code:

```python
ks = range(1, 6)
inertias = []

for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)
    
    # Fit model to samples
    model.fit(samples)
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()
```

This code plots an elbow, the value for the number of clusters is chosen at a point where the elbow starts to decrease.
Once the optimal number of clusters has been determined, we go to build the model:
```python
model = KMeans(n_clusters=3) # let's assume there are three clusters
labels = model.fit_predict(samples)

# Using crosstabulation to compare the original labels with the clusters' labels:
df = pd.DataFrame({'labels': labels, 'variesties':variesties}) # variesties is the orginal list of labels
ct = pd.crosstab(df['labels'], df['varieties'])
print(ct)
```

### Transforming Features for Better Clustering
For features scaling to Standardized data, where the mean of each feature is 0, and the variance becomes 1, we use **StandardScaler()**.
However, if we have to scale records/samples in our data, then we use **Normalizer()**.

```python
scaler = StandardScaler()
model = KMeans(n_clusters=3)
pipeline = make_pipeline(scaler, model)
pipeline.fit(samples)
cluster_labels = pipeline.predict(samples)
```
### Data Normalization

```python
from sklearn.preprocessing import Normalizer
normalizer = Normalizer()
kmeans = KMeans(n_clusters=10)
pipeline = make_pipeline(normalizer, kmeans)
pipeline.fit(movements)
```








