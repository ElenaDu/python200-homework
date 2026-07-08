#Build a Prefect pipeline that performs an end-to-end analysis of the World Happiness dataset.
import os
import pandas as pd
from prefect import task, flow
from prefect.logging import get_run_logger
import matplotlib.pyplot as plt
import seaborn as sns

#Task 1: Load Multiple Years of Data
@task(retries=3, retry_delay_seconds=2)
def load_data():
    logger = get_run_logger()

    current_folder = os.path.dirname(__file__)
    data_folder = os.path.join(current_folder, "happiness_project")
    output_folder = os.path.join(current_folder, "outputs")

    all_data = []
    for year in range(2015, 2025):
        file_path = os.path.join(data_folder, f"world_happiness_{year}.csv")
        df = pd.read_csv(file_path, sep=";", decimal=",")

        # Standardize the happiness score column name
        if "Ladder score" in df.columns:
            df.rename(
                columns={"Ladder score": "Happiness score"},
                inplace=True
            )

        df["Year"] = year
        all_data.append(df)
    
    # Merge all years into one DataFrame
    merged_df = pd.concat(all_data, ignore_index=True)

    # Save merged dataset
    merged_df.to_csv(
        os.path.join(output_folder, "merged_happiness.csv"),
        index=False
    )

    logger.info("Merged dataset saved successfully.")

    return merged_df
            
#Task 2: Descriptive Statistics 
@task
def descriptive_statistics(df):
    logger = get_run_logger()

    happiness_mean = df["Happiness score"].mean()
    logger.info(f"Mean happiness score: {happiness_mean:.2f}")

    happiness_median = df["Happiness score"].median()
    logger.info(f"Median happiness score: {happiness_median:.2f}")

    happiness_std = df["Happiness score"].std()
    logger.info(f"Standard deviation: {happiness_std:.2f}")

    mean_by_year = df.groupby("Year")["Happiness score"].mean()
    logger.info(f"Mean happiness score by year:\n{mean_by_year}")

    mean_by_region = df.groupby("Regional indicator")["Happiness score"].mean()
    logger.info(f"Mean happiness score by region:\n{mean_by_region}")


#Task 3: Visual Exploration
@task
def visual_exploration(df):
    logger = get_run_logger()
    current_folder = os.path.dirname(__file__)
    output_folder = os.path.join(current_folder, "outputs")

    #Histogram of all happiness scores across all years
    plt.figure(figsize=(8, 5))
    plt.hist(df["Happiness score"], bins=20, edgecolor="black")
    plt.title("Happiness Score Distribution Across All Years")
    plt.xlabel("Happiness Score")
    plt.ylabel("Frequency")

    plt.savefig(os.path.join(output_folder, "happiness_histogram.png"))
    plt.close()

    logger.info("Saved happiness_histogram.png")

    #Boxplot comparing happiness score distributions across years (one box per year).
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="Year", y="Happiness score")
    plt.title("Happiness Scores by Year")
    plt.xlabel("Year")
    plt.ylabel("Happiness Score")

    plt.tight_layout()

    plt.savefig(os.path.join(output_folder, "happiness_by_year.png"))
    plt.close()

    logger.info("Saved happiness_by_year.png")


    #Scatter plot showing the relationship between GDP per capita and happiness score.
    plt.figure(figsize=(8, 6))
    plt.scatter(df["GDP per capita"], df["Happiness score"])
    plt.title("GDP per Capita vs. Happiness Score")
    plt.xlabel("GDP per Capita")
    plt.ylabel("Happiness Score")

    plt.tight_layout()

    plt.savefig(os.path.join(output_folder, "gdp_vs_happiness.png"))
    plt.close()

    logger.info("Saved gdp_vs_happiness.png")
    
    
    






@flow
def happiness_pipeline():
    df = load_data()
    descriptive_statistics(df)
    visual_exploration(df)

           
if __name__ == "__main__":
    happiness_pipeline()

       