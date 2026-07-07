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
random_array = np.random.normal(65, 10, 500)

#Plot a histogram with 20 bins.
plt.hist(random_array, bins=20, color="skyblue", edgecolor="black")
plt.title("Distribution of Scores")
plt.xlabel("Value") 
plt.ylabel("Frequency")
plt.show()

# Descriptive Statistics Q3
group_a = [55, 60, 63, 70, 68, 62, 58, 65]
group_b = [75, 80, 78, 90, 85, 79, 82, 88]

#Create a boxplot comparing the two groups
plt.boxplot([group_a, group_b], labels=["Group A", "Group B"])
plt.title("Score Comparison")
plt.ylabel("Score")
plt.show()

# Descriptive Statistics Q4
normal_data = np.random.normal(50, 5, 200)
skewed_data = np.random.exponential(10, 200)

#Create side-by-side boxplots comparing the two distributions.
plt.boxplot([normal_data, skewed_data], labels=["Normal", "Exponential"])
plt.title("Distribution Comparison")
plt.ylabel("Value")
plt.show()

# The normal distribution appears approximately symmetric. 
# The exponential distribution is right-skewed: the upper whisker is much longer than the lower whisker, and there are numerous outliers.
# The mean is a good measure of central tendency for the normal distribution. For the exponential distribution, the median is a better measure because
# it is less affected by skewness and outliers.


# Descriptive Statistics Q5
data1 = [10, 12, 12, 16, 18]
data2 = [10, 12, 12, 16, 150]
from statistics import mode

print("Data1 measures of central tendency: ")
print(f"Mean: {np.mean(data1)}")
print(f"Median: {np.median(data1)}")
print(f"Mode: {mode(data1)}")

print("Data2 measures of central tendency: ")
print(f"Mean: {np.mean(data2)}")
print(f"Median: {np.median(data2)}")
print(f"Mode: {mode(data2)}")

# The value 150 from data2 is an outlier.
# The outlier makes the mean much larger, but it has little effect on the median.



# --- Hypothesis Testing ---

# Hypothesis Testing Q1
from scipy import stats
group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]

#Run an independent samples t-test on the two groups below. Print the t-statistic and p-value.

t_stat, p_val = stats.ttest_ind(group_a, group_b)

print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_val:.6f}")


# Hypothesis Testing Q2

if p_val < 0.05:
    print("The difference is statistically significant.")
else:
    print("No statistically significant difference detected.")

# Hypothesis Testing Q3
#Run a paired t-test on the before/after scores below (the same students measured twice). Print the t-statistic and p-value.
before = [60, 65, 70, 58, 62, 67, 63, 66]
after  = [68, 70, 76, 65, 69, 72, 70, 71]

t_stat, p_val = stats.ttest_rel(before, after)

print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_val:.6f}")

# Hypothesis Testing Q4
# Run a one-sample t-test to check whether the mean of scores is significantly different from a national benchmark of 70. 
# Print the t-statistic and p-value.
scores = [72, 68, 75, 70, 69, 74, 71, 73]

t_stat, p_val = stats.ttest_1samp(scores, 70)
print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_val:.6f}")


# Hypothesis Testing Q5
# Re-run the test from Q1 as a one-tailed test to check whether group_a scores are less than group_b scores. 
# Print the resulting p-value. Use the alternative parameter.

# One-tailed: is class_a less than class_b?
t_stat, p_val = stats.ttest_ind(group_a, group_b, alternative="less")
print(f"p-value: {p_val:.6f}")

# Hypothesis Testing Q6

print(
    "Conclusion: Group B had a higher average score than Group A. "
    f"The p-value was {p_val:.8f}, which is below 0.05, "
    "so the difference is unlikely to be due to chance."
)

# --- Correlation ---

# Correlation Q1
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
corr_matrix = np.corrcoef(x, y)
print("Correlation matrix:")
print(corr_matrix)
print(f"Correlation coefficient: {corr_matrix[0, 1]:.3f}")

# I expect the correlation to be 1 because y increases as x increases. This is a positive correlation.

# Correlation Q2
from scipy.stats import pearsonr

x = [1,  2,  3,  4,  5,  6,  7,  8,  9, 10]
y = [10, 9,  7,  8,  6,  5,  3,  4,  2,  1]

r, p = pearsonr(x,y)
print(f"Correlation coefficient: {r:.2f}")
print(f"P-value: {p}")

# Correlation Q3
people = {
    "height": [160, 165, 170, 175, 180],
    "weight": [55,  60,  65,  72,  80],
    "age":    [25,  30,  22,  35,  28]
}
df = pd.DataFrame(people)
print("Correlation matrix:")
corr_matrix = df.corr()
print(corr_matrix)

# Correlation Q4
x = [10, 20, 30, 40, 50]
y = [90, 75, 60, 45, 30]
plt.scatter(x, y, color="green")
plt.title("Negative Correlation")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# Correlation Q5
import seaborn as sns
sns.heatmap(corr_matrix, annot=True)
plt.title("Correlation Heatmap")
plt.show()

# --- Pipelines---

# Pipeline Q1
arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

# Function takes a NumPy array and returns a pandas Series with the name "values".
def create_series(arr):
    return pd.Series(arr, name="values")

# Function takes the Series, removes any NaN values using .dropna(), and returns the cleaned Series.
def clean_data(series):
    return series.dropna()

# Function takes the cleaned Series and returns a dictionary with four keys.
def summarize_data(series):
    return {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0]
    }

# Function calls the three functions above in sequence and returns the summary dictionary.
def data_pipeline(arr):
    series = create_series(arr)
    cleaned_series = clean_data(series)
    summary = summarize_data(cleaned_series)
    return summary

result = data_pipeline(arr)

for key, value in result.items():
    print(f"{key}: {value}")
    