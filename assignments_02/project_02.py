# --- Mini-Project -- Predicting Student Math Performance --- 
import pandas as pd
import os
import matplotlib.pyplot as plt

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
