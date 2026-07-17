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


# --- Linear Regression --- 
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

np.random.seed(42)
num_patients = 100
age    = np.random.randint(20, 65, num_patients).astype(float)
smoker = np.random.randint(0, 2, num_patients).astype(float)
cost   = 200 * age + 15000 * smoker + np.random.normal(0, 3000, num_patients)

# LR Q1
plt.scatter(age, cost, c=smoker, cmap="coolwarm")
plt.title("Medical Cost vs Age")
plt.xlabel("Age")
plt.ylabel("Medical Cost")
plt.savefig("outputs/cost_vs_age.png")
plt.show()

# The scatter plot shows two distinct groups. The upper group represents smokers, who generally have higher medical costs, 
# while the lower group represents non-smokers. This suggests that smoker status has a strong effect on medical cost.

# LR Q2
X = age.reshape(-1, 1)
y = cost

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# LR Q3

# 1. Create the Linear Regression model
model = LinearRegression()

# 2. Fit the model
model.fit(X_train, y_train)

# 3. Predict on the test set
y_pred = model.predict(X_test)

rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
print("RMSE:", rmse)

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

r2_age = model.score(X_test, y_test)
print("R² on the test set:", r2_age)

# The slope represents the estimated increase in annual medical cost for each additional year of age.
# A positive slope means that, on average, medical costs increase as people get older.


# LR Q4
X_full = np.column_stack([age, smoker])
y = cost

X_train, X_test, y_train, y_test = train_test_split( X_full, y, test_size=0.2, random_state=42)

# 1. Create the Linear Regression model
model_full = LinearRegression()

# 2. Fit the model
model_full.fit(X_train, y_train)

# 3. Predict on the test set
y_pred = model_full.predict(X_test)

r2_full = model_full.score(X_test, y_test)

print("R² using age only (Model 1): ", r2_age)
print("R² using age and smoker (Model 2): ", r2_full)

print("Age coefficient:", model_full.coef_[0])
print("Smoker coefficient:", model_full.coef_[1])

# Adding the smoker feature greatly improves the model because the test R² increases from about 0.07 to 0.77.
# The smoker coefficient means that, on average, smokers are predicted to have about $14,538 higher annual medical costs than non-smokers of the same age.


# LR Q5

plt.scatter(y_pred, y_test)
min_value = min(y_pred.min(), y_test.min())
max_value = max(y_pred.max(), y_test.max())
plt.plot([min_value, max_value], [min_value, max_value], color="red")
plt.title("Predicted vs Actual")
plt.xlabel("Predicted Medical Cost")
plt.ylabel("Actual Medical Cost")
plt.savefig("outputs/predicted_vs_actual.png")
plt.show()

# Points above the diagonal have actual medical costs that are higher than the model predicted, meaning the model underestimated the cost.
# Points below the diagonal have actual medical costs that are lower than the model predicted, meaning the model overestimated the cost.
