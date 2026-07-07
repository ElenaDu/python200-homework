# --- Pipelines---

# Pipeline Q2
import numpy as np
import pandas as pd
from prefect import task, flow

arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

# Define tasks
@task
def create_series(arr):
    return pd.Series(arr, name="values")

@task
def clean_data(series):
    return series.dropna()

@task
def summarize_data(series):
    return {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0]
    }

@flow
def pipeline_flow():
    series = create_series(arr)
    cleaned_series = clean_data(series)
    summary = summarize_data(cleaned_series)

    for key, value in summary.items():
        print(f"{key}: {value}")

    return summary

if __name__ == "__main__":
    pipeline_flow()


# Questions:
# 1. This pipeline is very small and only works with a few numbers. Using Prefect here feels like extra work because regular Python   
#    functions can do the same job with less code.
#
# 2. Prefect becomes more useful when working with bigger projects.
#    For example, it can help run pipelines on a schedule, process large amounts of data, connect several steps together, and make it easier   
#    to see if something fails and needs to be rerun.