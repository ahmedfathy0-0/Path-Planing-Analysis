import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, f_oneway
from scipy.stats import ttest_ind


# Load static and dynamic data
static_astar = pd.read_csv("Astar.csv")  # Replace with actual file path
dynamic_astar = pd.read_csv("simulatedastar.csv")  # Replace with actual file path
static_dijkstra = pd.read_csv("D.csv")  # Replace with actual file path
dynamic_dijkstra = pd.read_csv("simulatedijkstra.csv")  # Replace with actual file path
static_astar = static_astar.head(300)
static_dijkstra = static_dijkstra.head(300)

# For static algorithms: Extract columns and calculate Computational Efficiency
static_astar['Computational Efficiency'] = static_astar['Visited'] / static_astar['Execution Time']
static_dijkstra['Computational Efficiency'] = static_dijkstra['Visited'] / static_dijkstra['Execution Time']

# For dynamic algorithms: Extract columns and calculate Computational Efficiency
dynamic_astar['Computational Efficiency'] = dynamic_astar['Total visited'] / dynamic_astar['Execution Time']
dynamic_dijkstra['Computational Efficiency'] = dynamic_dijkstra['Total visited'] / dynamic_dijkstra['Execution Time']

# For static algorithms: Path Length and Computational Efficiency
static_astar_path_lengths = static_astar['Path Length']
static_dijkstra_path_lengths = static_dijkstra['Path Length']
static_astar_efficiency = static_astar['Computational Efficiency']
static_dijkstra_efficiency = static_dijkstra['Computational Efficiency']

# For dynamic algorithms: Path Length and Computational Efficiency
dynamic_astar_path_lengths = dynamic_astar['Actual path']
dynamic_dijkstra_path_lengths = dynamic_dijkstra['Actual path']
dynamic_astar_efficiency = dynamic_astar['Computational Efficiency']
dynamic_dijkstra_efficiency = dynamic_dijkstra['Computational Efficiency']

# Combine data for static and dynamic algorithms
static_path_lengths = pd.concat([static_astar_path_lengths, static_dijkstra_path_lengths], axis=0, ignore_index=True)
dynamic_path_lengths = pd.concat([dynamic_astar_path_lengths, dynamic_dijkstra_path_lengths], axis=0, ignore_index=True)
static_efficiency = pd.concat([static_astar_efficiency, static_dijkstra_efficiency], axis=0, ignore_index=True)
dynamic_efficiency = pd.concat([dynamic_astar_efficiency, dynamic_dijkstra_efficiency], axis=0, ignore_index=True)

# Descriptive statistics for central tendency (mean, median) and variability (variance, standard deviation)
def descriptive_stats(data, label):
    print(f"Descriptive Statistics for {label}:")
    print(f"Mean: {data.mean():.4f}")
    print(f"Median: {data.median():.4f}")
    print(f"Variance: {data.var():.4f}")
    print(f"Standard Deviation: {data.std():.4f}")
    print()

# Calculate for Path Lengths (both static and dynamic)
descriptive_stats(static_path_lengths, "Static Algorithms - Path Lengths")
descriptive_stats(dynamic_path_lengths, "Dynamic Algorithms - Path Lengths")

# Calculate for Computational Efficiency (both static and dynamic)
descriptive_stats(static_efficiency, "Static Algorithms - Computational Efficiency")
descriptive_stats(dynamic_efficiency, "Dynamic Algorithms - Computational Efficiency")

# Visualizing the distributions (Histograms for Path Length)
plt.figure(figsize=(14, 6))
plt.subplot(2, 2, 1)
plt.hist(static_path_lengths, bins=20, color='blue', alpha=0.7)
plt.xlim(80, 120)  # Adjust the x-axis limits as needed
plt.title('Static Algorithms - Path Length Distribution')
plt.xlabel('Path Length')
plt.ylabel('Frequency')

# Histogram for Dynamic Algorithms (Path Length)
plt.subplot(2, 2, 3)
plt.hist(dynamic_path_lengths, bins=20, color='green', alpha=0.7)
plt.xlim(80, 120)  # Adjust the x-axis limits as needed
plt.title('Dynamic Algorithms - Path Length Distribution')
plt.xlabel('Path Length')
plt.ylabel('Frequency')


# Histogram for Static Algorithms (Computational Efficiency)
plt.tight_layout()
plt.show()

# Scatter plot: Relationship between obstacle density and path length for both static and dynamic data
plt.scatter(static_astar['Density'], static_astar['Path Length'], label='Static A*', alpha=0.5, color='blue')
plt.scatter(dynamic_astar['Density'], dynamic_astar['Actual path'], label='Dynamic A*', alpha=0.5, color='green')

# Customizing the axis limits (scale adjustment)
plt.xlim(0.3, 0.8)  # Change this to the appropriate range for your data
plt.ylim(85, 130)  # Change this to the appropriate range for your data
plt.yscale('log')  # Logarithmic scale for better visualization
plt.xscale('log')
plt.title('Relationship between Obstacle Density and Path Length')
plt.xlabel('Obstacle Density')
plt.ylabel('Path Length')

plt.legend()
plt.show()

# Correlation: Pearson correlation coefficient between obstacle density and path length
static_corr, _ = pearsonr(static_astar['Density'], static_astar['Path Length'])
dynamic_corr, _ = pearsonr(dynamic_astar['Density'], dynamic_astar['Actual path'])
print(f"Pearson correlation (Static A*): {static_corr:.4f}")
print(f"Pearson correlation (Dynamic A*): {dynamic_corr:.4f}")

# Hypothesis Testing (One-Way ANOVA) for Computational Efficiency between Static and Dynamic Algorithms
f_statistic, p_value = f_oneway(static_efficiency, dynamic_efficiency)

# Output results
alpha = 0.05  # Significance level
print("One-Way ANOVA Results for Computational Efficiency:")
print(f"F-statistic: {f_statistic:.4f}")
print(f"P-value: {p_value:.4f}")

if p_value < alpha:
    print("Decision: Reject the null hypothesis.")
    print("There is a statistically significant difference in Computational Efficiency between static and dynamic algorithms.")
else:
    print("Decision: Fail to reject the null hypothesis.")
    print("There is no statistically significant difference in Computational Efficiency between static and dynamic algorithms.")


t_stat, p_value = ttest_ind(static_efficiency, dynamic_efficiency)

# Display the results
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")

# Interpretation
alpha = 0.05  # Significance level
if p_value < alpha:
    print("Reject H₀: There is a significant difference in Computational Efficiency between static and dynamic algorithms.")
else:
    print("Fail to reject H₀: There is no significant difference in Computational Efficiency.")

f_stat, p_value = f_oneway(static_data, dynamic_data)

# Output the results
print(f"F-statistic: {f_stat:.4f}")
print(f"P-value: {p_value:.4f}")

# Interpretation
alpha = 0.05
if p_value < alpha:
    print("Reject H₀: There is a significant difference in execution times across densities.")
else:
    print("Fail to reject H₀: There is no significant difference in execution times across densities.")
from sklearn.linear_model import LinearRegression
import numpy as np

# Assuming X contains the independent variables (Density and Path Length)
# and y contains the dependent variable (Computational Efficiency)
X = np.column_stack([density_data, path_length_data])
y = computational_efficiency_data

# Create the regression model
model = LinearRegression()
model.fit(X, y)

# Coefficients (beta values)
print(f"Intercept (β0): {model.intercept_}")
print(f"Coefficients (β1, β2): {model.coef_}")

# R-squared value for the goodness of fit
print(f"R-squared: {model.score(X, y)}")

# import numpy as np
# import pandas as pd
# from scipy.stats import ttest_ind, f_oneway
# from sklearn.linear_model import LinearRegression
# import matplotlib.pyplot as plt

# # Load the datasets
# static_astar = pd.read_csv("Astar.csv")  # Static A* data (Seed, Density, Execution Time, Path Length, Visited)
# dynamic_astar = pd.read_csv("simulatedastar.csv")  # Dynamic A* data (Seed, Density, Population, Actual path, Execution Time, Total visited)

# static_dijkstra = pd.read_csv("D.csv")  # Static Dijkstra data (Seed, Density, Execution Time, Path Length, Visited)
# dynamic_dijkstra = pd.read_csv("simulatedijkstra.csv")  # Dynamic Dijkstra data (Seed, Density, Population, Actual path, Execution Time, Total visited)

# # Calculate Computational Efficiency = Visited / Execution Time for static and dynamic
# static_astar['Computational Efficiency'] = static_astar['Visited'] / static_astar['Execution Time']
# dynamic_astar['Computational Efficiency'] = dynamic_astar['Total visited'] / dynamic_astar['Execution Time']

# static_dijkstra['Computational Efficiency'] = static_dijkstra['Visited'] / static_dijkstra['Execution Time']
# dynamic_dijkstra['Computational Efficiency'] = dynamic_dijkstra['Total visited'] / dynamic_dijkstra['Execution Time']

# # Extract relevant columns for analysis
# static_astar_efficiency = static_astar['Computational Efficiency']
# dynamic_astar_efficiency = dynamic_astar['Computational Efficiency']

# static_dijkstra_efficiency = static_dijkstra['Computational Efficiency']
# dynamic_dijkstra_efficiency = dynamic_dijkstra['Computational Efficiency']

# # 1. Hypothesis Testing: Independent t-test (Comparing Computational Efficiency between Static and Dynamic for A* and Dijkstra)
# # A* t-test
# t_stat_astar, p_value_astar = ttest_ind(static_astar_efficiency, dynamic_astar_efficiency)

# # Dijkstra t-test
# t_stat_dijkstra, p_value_dijkstra = ttest_ind(static_dijkstra_efficiency, dynamic_dijkstra_efficiency)

# # Output t-test results
# print("t-test results for A*:")
# print(f"t-statistic: {t_stat_astar:.4f}")
# print(f"p-value: {p_value_astar:.4f}")
# if p_value_astar < 0.05:
#     print("Reject H₀: There is a significant difference in Computational Efficiency between static and dynamic A* algorithms.")
# else:
#     print("Fail to reject H₀: There is no significant difference in Computational Efficiency for A*.")

# print("\nt-test results for Dijkstra:")
# print(f"t-statistic: {t_stat_dijkstra:.4f}")
# print(f"p-value: {p_value_dijkstra:.4f}")
# if p_value_dijkstra < 0.05:
#     print("Reject H₀: There is a significant difference in Computational Efficiency between static and dynamic Dijkstra algorithms.")
# else:
#     print("Fail to reject H₀: There is no significant difference in Computational Efficiency for Dijkstra.")

# # 2. ANOVA: Compare execution times across different obstacle densities for Static and Dynamic Algorithms (A* and Dijkstra)
# # A* Density Groups
# static_astar_density_groups = [group['Execution Time'].values for _, group in static_astar.groupby('Density')]
# dynamic_astar_density_groups = [group['Execution Time'].values for _, group in dynamic_astar.groupby('Density')]

# # Dijkstra Density Groups
# static_dijkstra_density_groups = [group['Execution Time'].values for _, group in static_dijkstra.groupby('Density')]
# dynamic_dijkstra_density_groups = [group['Execution Time'].values for _, group in dynamic_dijkstra.groupby('Density')]

# # Perform ANOVA for A*
# f_stat_astar, p_value_astar_anova = f_oneway(*static_astar_density_groups, *dynamic_astar_density_groups)
# # Perform ANOVA for Dijkstra
# f_stat_dijkstra, p_value_dijkstra_anova = f_oneway(*static_dijkstra_density_groups, *dynamic_dijkstra_density_groups)

# # Output ANOVA results
# print("\nANOVA results for A*:")
# print(f"F-statistic: {f_stat_astar:.4f}")
# print(f"P-value: {p_value_astar_anova:.4f}")
# if p_value_astar_anova < 0.05:
#     print("Reject H₀: There is a significant difference in execution times across densities for A*.")
# else:
#     print("Fail to reject H₀: There is no significant difference in execution times across densities for A*.")

# print("\nANOVA results for Dijkstra:")
# print(f"F-statistic: {f_stat_dijkstra:.4f}")
# print(f"P-value: {p_value_dijkstra_anova:.4f}")
# if p_value_dijkstra_anova < 0.05:
#     print("Reject H₀: There is a significant difference in execution times across densities for Dijkstra.")
# else:
#     print("Fail to reject H₀: There is no significant difference in execution times across densities for Dijkstra.")

# # 3. Regression Analysis: Multiple Linear Regression (Impact of Density and Path Length on Computational Efficiency)
# # Prepare data for regression (A* and Dijkstra)
# X_static_astar = static_astar[['Density', 'Path Length']]  # Independent variables: Density, Path Length
# y_static_astar = static_astar['Computational Efficiency']

# X_dynamic_astar = dynamic_astar[['Density', 'Actual path']]  # Independent variables: Density, Actual path
# y_dynamic_astar = dynamic_astar['Computational Efficiency']

# X_static_dijkstra = static_dijkstra[['Density', 'Path Length']]  # Independent variables: Density, Path Length
# y_static_dijkstra = static_dijkstra['Computational Efficiency']

# X_dynamic_dijkstra = dynamic_dijkstra[['Density', 'Actual path']]  # Independent variables: Density, Actual path
# y_dynamic_dijkstra = dynamic_dijkstra['Computational Efficiency']

# # Create regression model and fit it for A* and Dijkstra
# reg_static_astar = LinearRegression()
# reg_static_astar.fit(X_static_astar, y_static_astar)

# reg_dynamic_astar = LinearRegression()
# reg_dynamic_astar.fit(X_dynamic_astar, y_dynamic_astar)

# reg_static_dijkstra = LinearRegression()
# reg_static_dijkstra.fit(X_static_dijkstra, y_static_dijkstra)

# reg_dynamic_dijkstra = LinearRegression()
# reg_dynamic_dijkstra.fit(X_dynamic_dijkstra, y_dynamic_dijkstra)

# # Output regression results for A* Static
# print("\nStatic A* Regression Results:")
# print(f"Intercept: {reg_static_astar.intercept_}")
# print(f"Coefficients: {reg_static_astar.coef_}")
# print(f"R-squared: {reg_static_astar.score(X_static_astar, y_static_astar)}")

# # Output regression results for Dynamic A*
# print("\nDynamic A* Regression Results:")
# print(f"Intercept: {reg_dynamic_astar.intercept_}")
# print(f"Coefficients: {reg_dynamic_astar.coef_}")
# print(f"R-squared: {reg_dynamic_astar.score(X_dynamic_astar, y_dynamic_astar)}")

# # Output regression results for Static Dijkstra
# print("\nStatic Dijkstra Regression Results:")
# print(f"Intercept: {reg_static_dijkstra.intercept_}")
# print(f"Coefficients: {reg_static_dijkstra.coef_}")
# print(f"R-squared: {reg_static_dijkstra.score(X_static_dijkstra, y_static_dijkstra)}")

# # Output regression results for Dynamic Dijkstra
# print("\nDynamic Dijkstra Regression Results:")
# print(f"Intercept: {reg_dynamic_dijkstra.intercept_}")
# print(f"Coefficients: {reg_dynamic_dijkstra.coef_}")
# print(f"R-squared: {reg_dynamic_dijkstra.score(X_dynamic_dijkstra, y_dynamic_dijkstra)}")

# # 4. Plotting the Computational Efficiency distributions
# plt.figure(figsize=(14, 6))

# # Histogram for Static A* Computational Efficiency
# plt.subplot(2, 2, 1)
# plt.hist(static_astar_efficiency, bins=20, color='blue', alpha=0.7)
# plt.title('Static A* - Computational Efficiency')
# plt.xlabel('Computational Efficiency')
# plt.ylabel('Frequency')

# # Histogram for Dynamic A* Computational Efficiency
# plt.subplot(2, 2, 2)
# plt.hist(dynamic_astar_efficiency, bins=20, color='green', alpha=0.7)
# plt.title('Dynamic A* - Computational Efficiency')
# plt.xlabel('Computational Efficiency')
# plt.ylabel('Frequency')

# # Histogram for Static Dijkstra Computational Efficiency
# plt.subplot(2, 2, 3)
# plt.hist(static_dijkstra_efficiency, bins=20, color='red', alpha=0.7)
# plt.title('Static Dijkstra - Computational Efficiency')
# plt.xlabel('Computational Efficiency')
# plt.ylabel('Frequency')

# # Histogram for Dynamic Dijkstra Computational Efficiency
# plt.subplot(2, 2, 4)
# plt.hist(dynamic_dijkstra_efficiency, bins=20, color='orange', alpha=0.7)
# plt.title('Dynamic Dijkstra - Computational Efficiency')
# plt.xlabel('Computational Efficiency')
# plt.ylabel('Frequency')

# # Adjust the layout
# plt.tight_layout()
# plt.show()
