import pandas as pd
from scipy.stats import f_oneway

# Load the data for the three algorithms
# Replace with the actual file paths
rrt_data = pd.read_csv("RRT.csv").head(100)  # Take only the first 100 rows
astar_data = pd.read_csv("Astar.csv").head(100)  # Take only the first 100 rows
dijkstra_data = pd.read_csv("D.csv").head(100)  # Take only the first 100 rows

# # Extract the execution time column for each algorithm
rrt_times = rrt_data["Computational"]
astar_times = astar_data["Computational"]
dijkstra_times = dijkstra_data["Computational"]

# # Perform One-Way ANOVA
f_statistic, p_value = f_oneway(rrt_times, astar_times, dijkstra_times)

# Output results
alpha = 0.05  # Significance level
print("One-Way ANOVA Results:")
print(f"F-statistic: {f_statistic:.4f}")
print(f"P-value: {p_value:.4f}")

if p_value < alpha:
    print("Decision: Reject the null hypothesis.")
    print("There is a statistically significant difference in Computational Efficiency  between the algorithms.")
else:
    print("Decision: Fail to reject the null hypothesis.")
    print("There is no statistically significant difference in Computational Efficiency between the algorithms.")
