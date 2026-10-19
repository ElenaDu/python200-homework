# --- scikit-learn API ---

#Q1
import numpy as np
from sklearn.linear_model import LinearRegression

temp_min = np.array([1, 4, 8, 12, 16, 20]).reshape(-1, 1)
temp_max = np.array([8, 10, 15, 18, 23, 27])

# Create the model
model = LinearRegression()

# Fit the model to the data
model.fit(temp_min, temp_max)

# Predict the high temperatures for low temperatures of 6 and 18 degrees
predictions = model.predict(np.array([6, 18]).reshape(-1, 1))

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("Predicted high for a low of 6 degrees:", predictions[0])
print("Predicted high for a low of 18 degrees:", predictions[1])

#Q2
x = np.array([10, 20, 30, 40, 50])
print("Original shape:", x.shape)

x = x.reshape(-1, 1)
print("New shape:", x.shape)

# Scikit-learn needs X to be 2D because each row represents one example,
# and each column contains one piece of information about that example.
# Even if we have only one feature, we still need to organize the data
# into rows and columns so the model can understand it.


# --- Linear Regression ---

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

rng = np.random.default_rng(42)
num_days  = 120
temp_min  = rng.uniform(-5, 22, num_days)
is_rainy  = rng.integers(0, 2, num_days).astype(float)
temp_max  = 1.0 * temp_min + 6 - 4 * is_rainy + rng.normal(0, 2, num_days)


#Q1
from pathlib import Path
Path("outputs").mkdir(exist_ok=True)

# Create a scatter plot
plt.scatter(temp_min, temp_max, c=is_rainy, cmap="coolwarm")

plt.title("Daily High vs Low")
plt.xlabel("Low Temperature")
plt.ylabel("High Temperature")
plt.savefig("outputs/high_vs_low.png")
plt.show()

# The plot shows two groups of points: blue points represent dry days (is_rainy = 0),
# and red points represent rainy days (is_rainy = 1). For similar low temperatures,
# the red points generally have lower high temperatures than the blue points.
# This suggests that rainy days tend to have lower high temperatures.


#Q2
# Reshape temp_min into a 2D array for scikit-learn
X = temp_min.reshape(-1, 1)

# Use temp_max as the target
y = temp_max

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Print the shapes of all four arrays
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

#Q3
# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Print the slope and intercept
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

# Predict high temperatures for the test data
y_pred = model.predict(X_test)

# Calculate and print RMSE and R-squared
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", rmse)

print("R² on the test set:", model.score(X_test, y_test))

# The slope tells us how much the predicted daily high changes when the daily low temperature increases by one degree.
# Slope: 0.944 — for every 1°C increase in the daily low temperature, the model predicts that the daily high will increase by about 0.94°C.


#Q4

# Add is_rainy as a second feature
X_full = np.column_stack([temp_min, is_rainy])

# Split the data into training and test sets
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X_full, temp_max, test_size=0.2, random_state=42
)

# Create and train the new model
model_full = LinearRegression()
model_full.fit(X_train_full, y_train_full)

# Calculate the test R-squared
r2_full = model_full.score(X_test_full, y_test_full)
print("R² from Question 3 (low temperature only):", model.score(X_test, y_test))
print("R² from Question 4 (low temperature and rain):", r2_full)

# Print both coefficients
print("temp_min coefficient:", model_full.coef_[0])
print("is_rainy coefficient:", model_full.coef_[1])

# The is_rainy coefficient shows how much lower or higher the predicted daily high is on rainy days compared with dry days, when the low
# temperature stays the same.
# The is_rainy coefficient is about -2.92, which means that for the same low temperature, the predicted high is about 2.92 degrees lower
# on rainy days than on dry days.


#Q5
# Predict high temperatures using the two-feature model
y_pred_full = model_full.predict(X_test_full)

# Create the predicted vs actual plot
plt.scatter(y_pred_full, y_test_full)

# Add a diagonal reference line
plt.plot(
    [y_test_full.min(), y_test_full.max()],
    [y_test_full.min(), y_test_full.max()],
    "r--"
)

plt.title("Predicted vs Actual")
plt.xlabel("Predicted High Temperature")
plt.ylabel("Actual High Temperature")

# Save the plot
plt.savefig("outputs/predicted_vs_actual_warmup.png")
plt.show()

# Points above the diagonal mean the actual high was higher than predicted.
# Points below the diagonal mean the actual high was lower than predicted.
# In our plot, most points are fairly close to the red line, which suggests that the model's predictions
# are reasonably close to the actual high temperatures. 
# A few points are farther away, showing larger prediction errors.
