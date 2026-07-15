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