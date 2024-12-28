import pandas as pd
import numpy as np
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt

# Step 1: Load datasets (replace with actual file paths)
static_df = pd.read_csv("D.csv")  # Static environment dataset
dynamic_df = pd.read_csv("simulatedijkstra.csv")  # Dynamic environment dataset

# Step 2: Merge datasets on Seed and Density
merged_df = pd.merge(
    dynamic_df,
    static_df,
    on=["Seed", "Density"],
    suffixes=("_dynamic", "_static")
)

# Step 3: Analyze Metrics for Each Population
results = []
metrics = ["Execution Time", "Path Length", "Visited"]

for population in dynamic_df["Population"].unique():
    subset = merged_df[merged_df["Population"] == population]

    # Perform paired t-tests for each metric
    for metric in metrics:
        dynamic_metric = subset[f"{metric}_dynamic"].values
        static_metric = subset[f"{metric}_static"].values

        t_stat, p_value = ttest_rel(static_metric, dynamic_metric)

        # Store results
        results.append({
            "Population": population,
            "Metric": metric,
            "T-Statistic": t_stat,
            "P-Value": p_value
        })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Step 4: Visualize Differences for Each Metric
for metric in metrics:
    metric_diff = merged_df.groupby("Population").apply(
        lambda g: np.mean(g[f"{metric}_dynamic"] - g[f"{metric}_static"])
    ).reset_index(name="Difference")

    plt.plot(metric_diff["Population"], metric_diff["Difference"], label=metric)

plt.xlabel("Population (Moving Obstacles)")
plt.ylabel("Mean Difference (Dynamic - Static)")
plt.title("Performance Metric Differences by Population")
plt.legend()
plt.show()

# Step 5: Save Results
results_df.to_csv("path_planning_ttest_results.csv", index=False)
print("Analysis complete. Results saved to 'path_planning_ttest_results.csv'.")