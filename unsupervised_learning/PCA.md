**PCA for Feature Engineering**
There are two ways to use PCA for feature engineering.

1) As a descriptive technique:
   Since the components tell about the variation, we could compute the MI scores for the components and see what kind of variation is most predictive of the target.
   That could give us ideas for kinds of features to create. For example, a product of 'Height' and 'Diameter' if 'Size' is important, or a ratio of 'Height' and 'Diameter'
   if Shape is important.
   
2) Using the components themselves as features:
   Because the components expose the variational structure of the data directly, they can often be more informative than the original features.


**PCA use cases:** 
- Dimensionality reduction, used when multiple features are highly correlated.
- Anomaly detection, to identify unusual data points (outliers).
- Noise Cancellation, In image processing, PCA can reduce noise in pictures while preserving the important visual information.
- Decorrelation 

**What are redundant features?**
When two or more features are highly correlated, they contain similar information. This overlap means that one feature can be predicted from the other with little error.
_Example:_ In a dataset with both "height" and "arm span" of individuals, these two features might be highly correlated because taller individuals typically have larger arm spans. 
Including both doesn't add much new information.

**Why Redundancy is a Problem?**
- Inefficiency: More features increase computational cost and complexity without adding new information.
- Overfitting: Models may learn noise or irrelevant patterns, reducing their generalizability to new data.
- Interpretability: Having too many similar features can make the model harder to interpret.

**Steps in PCA for Removing Redundancy:**
- Standardize the Data: Ensure each feature contributes equally by scaling.
- Compute Principal Components: Use linear algebra techniques to find the principal components.
- Analyze Variance: Check the explained variance ratio to see how much information each component captures.
- Select Components: Keep components that capture significant variance (e.g., 95% of total variance) and drop those that do not.





