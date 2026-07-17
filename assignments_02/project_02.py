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
plt.figure(figsize=(8, 6))

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
