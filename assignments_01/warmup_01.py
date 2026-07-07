# --- Pandas ---

# Pandas Q1
import pandas as pd
data = {
    "name":   ["Alice", "Bob", "Carol", "David", "Eve"],
    "grade":  [85, 72, 90, 68, 95],
    "city":   ["Boston", "Austin", "Boston", "Denver", "Austin"],
    "passed": [True, True, True, False, True]
}

df = pd.DataFrame(data)
print(f"First three rows: {df.head(3)}")
print(f"DataFrame shape: {df.shape}")
print(f"Data types: {df.dtypes}")

# Pandas Q2
select_students = df[(df["passed"]==True)& (df["grade"]>80)]
print(f"Students who passed and have a grade above 80: {select_students}")

# Pandas Q3
df["grade_curved"] = df["grade"]+5
print(f"DataFrame with a new grade_curved column: {df}")

# Pandas Q4
df["name_upper"] = df["name"].str.upper()
print(f"Print name and name_upper columns:{df[['name', 'name_upper']]}")

# Pandas Q5
df.groupby("city")["grade"].mean()
print(f"The mean grade for each city: {df.groupby('city')['grade'].mean()}")

# Pandas Q6
print(df[['name', 'city']])
df['city'] = df['city'].replace('Austin', 'Houston')
print(f"Checking the 'name' and 'city' columns to confirm the change: {df[['name', 'city']]}")

# Pandas Q7
sorted_df = df.sort_values(by='grade', ascending=False)
print(f"Top 3 rows of the sorted dataframe: {sorted_df.head(3)}")


# --- NumPy ---
import numpy as np

# NumPy Q1
arr = np.array([10, 20, 30, 40, 50])
print(f"Shape: {arr.shape}")
print(f"Number of dimension: {arr.ndim}")
print(f"Data type of elements: {arr.dtype}")

# NumPy Q2
arr=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"Shape: {arr.shape}")
print(f"Size: {arr.size}")

# NumPy Q3
print(f"Top-left 2x2 block: {arr[:2, :2]}")

# NumPy Q4
array_0 = np.zeros((3,4))
print(f"3x4 array of zeros: {array_0}")
array_1 = np.ones((2,5))
print(f"2x5 array of ones: {array_1}")

# NumPy Q5
arr_q5 = np.arange(0, 50, 5)
print(f"Array: {arr_q5}")
print(f"Shape: {arr_q5.shape}")
print(f"Mean: {arr_q5.mean()}")
print(f"Sum: {arr_q5.sum()}")
print(f"Standard Deviation: {arr_q5.std():.2f}")

# NumPy Q6
arr_random_values = np.random.normal(loc=0.0, scale=1.0, size=200)
print(f"Mean: {arr_random_values.mean()}")
print(f"Standard deviation: {arr_random_values.std():.2f}")

# --- Matplotlib ---
import matplotlib.pyplot as plt

# Matplotlib Q1
x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]

plt.plot(x,y)
plt.title("Squares")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# Matplotlib Q2
subjects = ["Math", "Science", "English", "History"]
scores   = [88, 92, 75, 83]

plt.bar(subjects, scores, color = "orange")
plt.title("Subject Scores")
plt.xlabel("subjects")
plt.ylabel("scores")
plt.show()

# Matplotlib Q3
x1, y1 = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]
x2, y2 = [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]

plt.scatter(x1, y1, color="green", label="Dataset 1")
plt.scatter(x2, y2, color="red", label="Dataset 2")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()

# Matplotlib Q4

#Dataset from Q1:
x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]

#Dataset from Q2:
subjects = ["Math", "Science", "English", "History"]
scores   = [88, 92, 75, 83]

fig, ax = plt.subplots(1,2)  # Create a figure and axis

#Line plot
ax[0].plot(x, y)
ax[0].set_title("Squares")
ax[0].set_xlabel("x")
ax[0].set_ylabel("y")

#Bar plot
ax[1].bar(subjects, scores)
ax[1].set_title("Subject Scores")
ax[1].set_xlabel("Subjects")
ax[1].set_ylabel("Scores")

plt.tight_layout()
plt.show()



# --- Descriptive Statistics ---

# Descriptive Statistics Q1
data = [12, 15, 14, 10, 18, 22, 13, 16, 14, 15]
print(f"Mean: {np.mean(data)}")
print(f"Median: {np.median(data)}")
print(f"Variance: {np.var(data):.2f}")
print(f"Standard Deviation: {np.std(data):.2f}")

# Descriptive Statistics Q2

