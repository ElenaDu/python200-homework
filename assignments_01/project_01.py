#Build a Prefect pipeline that performs an end-to-end analysis of the World Happiness dataset.
import os
import pandas as pd
from prefect import task, flow
from prefect.logging import get_run_logger
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

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

    #Correlation heatmap showing the Pearson correlations between all numeric columns.
    plt.figure(figsize=(10, 8))

    correlation_matrix = df.corr(numeric_only=True)
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f")

    plt.title("Correlation Heatmap")
    plt.tight_layout()

    plt.savefig(os.path.join(output_folder, "correlation_heatmap.png"))
    plt.close()

    logger.info("Saved correlation_heatmap.png")
    
    
#Task 4: Hypothesis Testing
@task
def hypothesis_testing(df):
    logger = get_run_logger()

    #Run an independent samples t-test comparing happiness scores from 2019 to 2020.

    happiness_2019 = df[df["Year"] == 2019]["Happiness score"]
    happiness_2020 = df[df["Year"] == 2020]["Happiness score"]

    t_statistic, p_value = ttest_ind(happiness_2019, happiness_2020)

    mean_2019 = happiness_2019.mean()
    mean_2020 = happiness_2020.mean()

    logger.info(f"Mean happiness score (2019): {mean_2019:.2f}")
    logger.info(f"Mean happiness score (2020): {mean_2020:.2f}")

    logger.info(f"T-statistic: {t_statistic:.4f}")
    logger.info(f"P-value: {p_value:.4f}")

    if p_value < 0.05:
        logger.info("There is a statistically significant difference in the average happiness scores between 2019 and 2020."
                    "Based on this dataset, there is evidence that average happiness scores changed between these two years.")
    else:
        logger.info("There is no statistically significant difference in the average happiness scores between 2019 and 2020."
                    "Based on this dataset, we do not have enough evidence to conclude that the pandemic changed global happiness scores during that period.")
        
    

    #Run an independent samples t-test comparing happiness scores in North America and ANZ vs Central and Eastern Europe

    north_america_anz = df[df["Regional indicator"] == "North America and ANZ"]["Happiness score"]
    central_eastern_europe = df[df["Regional indicator"] == "Central and Eastern Europe"]["Happiness score"]

    t_statistic_regions, p_value_regions = ttest_ind(north_america_anz, central_eastern_europe)
    
    mean_north_america = north_america_anz.mean()
    mean_central_eastern_europe = central_eastern_europe.mean()

    logger.info(f"Mean happiness score (North America and ANZ): {mean_north_america:.2f}")
    logger.info(f"Mean happiness score (Central and Eastern Europe): {mean_central_eastern_europe:.2f}")

    logger.info(f"T-statistic (regions): {t_statistic_regions:.4f}")
    logger.info(f"P-value (regions): {p_value_regions:.4f}")

    if p_value_regions < 0.05:
        
        logger.info("There is a statistically significant difference in the average happiness scores between North America and ANZ and Central and Eastern Europe. "
        "Based on this dataset, there is evidence that average happiness scores differ between these two regions.")
    
    else:
        
        logger.info("There is no statistically significant difference in the average happiness scores between North America and ANZ and Central and Eastern Europe."
                    "Based on this dataset, we do not have enough evidence to conclude that the average happiness scores differ between these two regions.")
    
    
        
        
#Task 5: Correlation and Multiple Comparisons

    
   



@flow
def happiness_pipeline():
    df = load_data()
    descriptive_statistics(df)
    visual_exploration(df)
    hypothesis_testing(df)

           
if __name__ == "__main__":
    happiness_pipeline()

       