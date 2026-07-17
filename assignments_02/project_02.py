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



