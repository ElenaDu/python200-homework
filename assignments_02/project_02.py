# --- Mini-Project -- Predicting Student Math Performance --- 
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

os.makedirs("outputs", exist_ok=True)


# Task 1: Load and Explore

# The CSV file uses semicolons (;) as the delimiter, so sep=";" is required.
df = pd.read_csv("student_performance_math.csv", sep=";")

print("Dataset shape:\n", df.shape)
print("\nFirst five rows:\n", df.head())
print("\nData types:\n", df.dtypes)

plt.figure(figsize=(8, 5))
plt.hist(df["G3"], bins=21)
plt.title("Distribution of Final Math Grades")
plt.xlabel("Final Grade (G3)")
plt.ylabel("Number of Students")

plt.savefig("outputs/g3_distribution.png")
plt.show()

# Task 2: Preprocess the Data

# Students with G3 = 0 did not take the final exam.
# Removing these rows prevents the model from treating an absent student as someone who actually earned a score of zero.

df_filtered = df[df["G3"] != 0].copy()

print("Original dataset shape:\n", df.shape)
print("\nFiltered dataset shape:\n", df_filtered.shape)

# Convert yes/no columns to 1/0
binary_columns = ["schoolsup", "internet", "higher", "activities"]

for column in binary_columns:
    df_filtered[column] = df_filtered[column].map({"yes": 1, "no": 0})

# Convert sex: F=0, M=1
df_filtered["sex"] = df_filtered["sex"].map({"F": 0, "M": 1})

# Pearson correlation
correlation_original = df["absences"].corr(df["G3"])
correlation_filtered = df_filtered["absences"].corr(df_filtered["G3"])

print("Pearson correlation (original):", correlation_original)
print("Pearson correlation (filtered):", correlation_filtered)

# Students with G3 = 0 did not take the final exam, so their grade does not reflect their academic performance. 
# Many of these students also had a large number of absences. Including them mixes exam absences with actual exam scores,
# which masks the true relationship between absences and performance. After removing these rows, the negative correlation becomes stronger because G3
# represents only students who actually completed the exam.


# Task 3: Exploratory Data Analysis

numeric_columns = df_filtered.select_dtypes(include="number").columns.drop(["G1", "G2", "G3"])

# Pearson correlations with G3
correlations = df_filtered[numeric_columns].corrwith(df_filtered["G3"])

correlations = correlations.sort_values()
print("Pearson correlations with G3:\n", correlations)

#The feature with the strongest relationship to G3 is failures, with a Pearson correlation of -0.294. 
#This indicates that students with more past failures tend to have lower final math grades. 
# The positive correlations for Medu and Fedu suggest that students whose parents have higher education levels tend to earn slightly higher grades. 
# One surprising result is that absences has only a moderate negative correlation (-0.213), 
# indicating that attendance alone is not as strong a predictor of final grades as previous academic performance.

import seaborn as sns

# Heatmap of correlations
plt.figure(figsize=(10, 8))

heatmap_columns = list(numeric_columns) + ["G3"]
sns.heatmap(
    df_filtered[heatmap_columns].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("outputs/correlation_heatmap.png")
plt.show()
plt.close()

# The heatmap shows that failures has the strongest negative correlation with G3, while Medu and Fedu have the strongest positive correlations.
# Most other features have relatively weak relationships with the final grade.

# Box plot: Study Time vs. G3
df_filtered.boxplot(column="G3", by="studytime")

plt.title("Final Math Grade by Study Time")
plt.suptitle("")  # Remove the default pandas title
plt.xlabel("Weekly Study Time")
plt.ylabel("Final Math Grade (G3)")

plt.tight_layout()
plt.savefig("outputs/studytime_vs_g3_boxplot.png")
plt.show()
plt.close()

# The box plot shows that students who study more tend to have slightly higher final math grades.
# However, there is substantial overlap between the groups, indicating that study time alone is not a strong predictor
# of final performance.

# Use failures to predict G3
X = df_filtered[["failures"]]
y = df_filtered["G3"]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate evaluation metrics
slope = model.coef_[0]
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Slope:", round(slope, 3))
print("RMSE:", round(rmse, 3))
print("R²:", round(r2, 3))

# The slope of -1.428 means that each additional previous failure is associated with about a 1.4-point decrease in the predicted final grade.
# The RMSE of 2.962 means the model's predictions are typically off by about 3 grade points on a 0–20 grading scale.
# The R² value of 0.089 is  about what I expected because the exploratory data analysis showed only a moderate relationship between failures and the final grade.


# Task 5: Build the Full Model

# Select features and target
feature_cols = [
    "age",
    "Medu",
    "Fedu",
    "traveltime",
    "studytime",
    "failures",
    "absences",
    "freetime",
    "goout",
    "Walc",
    "schoolsup",
    "internet",
    "higher",
    "activities",
    "sex"
]

X = df_filtered[feature_cols]
y = df_filtered["G3"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Evaluate the model
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

# Print results
print("Train R²:", round(train_r2, 3))
print("Test R²:", round(test_r2, 3))
print("RMSE:", round(rmse, 3))

print("\nFeature Coefficients:")
for name, coef in zip(feature_cols, model.coef_):
    print(name, ":", round(coef, 3))


# The largest coefficients are schoolsup (-2.263), internet (1.037),
# failures (-0.800), sex (0.402), goout (-0.313), studytime (0.311),
# and Walc (-0.268). The negative coefficient for schoolsup is the most
# surprising because school support is intended to help students.
# A likely explanation is that students who receive extra support are
# already struggling academically, so the variable reflects students'
# existing difficulties rather than the effect of the support itself.
# The positive coefficient for internet may reflect that students with
# internet access have better access to learning resources.
#
# The train R² (0.235) and test R² (0.263) are very close. In fact, the
# test R² is slightly higher, suggesting that the model generalizes well
# and does not appear to be overfitting.
#
# If I were deploying this model, I would keep features with larger
# coefficients, such as schoolsup, internet, failures, sex, goout,
# studytime, and Walc, because they appear to have the greatest influence
# on the predictions. I would consider dropping features with coefficients
# close to zero, such as freetime, activities, higher, absences,
# traveltime, Medu, and Fedu, since they contribute relatively little
# to the model.

#Task 6: Evaluate and Summarize