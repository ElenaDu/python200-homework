# --- scikit-learn API --- 
# Q1
import numpy as np
from sklearn.linear_model import LinearRegression

years  = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])

# 1. Create the model
model = LinearRegression()

# 2. Fit model to data
model.fit(years, salary)

#3. Predict with new data
salary_4 = model.predict([[4]]) 
salary_8 = model.predict([[8]])

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("Predicted salary for 4 years of experience:", round(salary_4[0]))
print("Predicted salary for 8 years of experience:", round(salary_8[0]))


# Q2
x = np.array([10, 20, 30, 40, 50])
print("Original shape of the 1D array:", x.shape)

x = x.reshape(-1, 1)
print("New shape of the 2D array :", x.shape)

# The scikit-learn design accepts a 2D array as input, where rows represent samples and columns represent features.
# Even if there is only one feature, it must be stored as a column in a 2D array.

# Q3
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X_clusters, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.8, random_state=7)

# 1. Create the KMeans model
kmeans = KMeans(n_clusters=3, random_state=42)

# 2. Fit model to data
kmeans.fit(X_clusters)

# 3. Predict the cluster labels
labels = kmeans.predict(X_clusters)

print("Cluster centers:", kmeans.cluster_centers_)
print("Number of points in each cluster:", np.bincount(labels))

plt.scatter(X_clusters[:, 0], X_clusters[:, 1], c=labels)
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    marker="X",
    color="black",
    s=200,
    label="Cluster Centers"
)

plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.savefig("outputs/kmeans_clusters.png")
plt.show()

