import scipy.stats as stats
import pandas as pd
import numpy as np
file1 = 'Astar.csv'  
file2 = 'D.csv'
file3 = 'RRT.csv'
# Read the data
algo1 = pd.read_csv(file1)
algo2 = pd.read_csv(file2)
algo3 = pd.read_csv(file3)
# Perform ANOVA for Execution Time across different algorithms
execution_time_anova = stats.f_oneway(algo1['Execution Time'], algo2['Execution Time'], algo3['Execution Time'])

# # Perform ANOVA for Path Optimality across different algorithms
 path_optimality_anova = stats.f_oneway(algo1['optimality'], algo2['optimality'], algo3['optimality'])

# Display p-values for both tests
print(f"ANOVA for Execution Time: p-value = {execution_time_anova.pvalue}")
print(f"ANOVA for Path Optimality: p-value = {path_optimality_anova.pvalue}")
def confidence_interval(data, confidence=0.95):
    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)
    n = len(data)
    critical_value = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin_of_error = critical_value * (std_dev / np.sqrt(n))
    return mean - margin_of_error, mean + margin_of_error

# # Calculate CI for Execution Time for each algorithm
# ci_algo1 = confidence_interval(algo1['Execution Time'])
# ci_algo2 = confidence_interval(algo2['Execution Time'])
# ci_algo3 = confidence_interval(algo3['Execution Time'])

# print(f"A* Algorithm 95% CI for Execution Time: {ci_algo1}")
# print(f"Dijkstra Algorithm 95% CI for Execution Time: {ci_algo2}")
# print(f"RRT Algorithm 95% CI for Execution Time: {ci_algo3}")
# import pandas as pd
# import scipy.stats as stats

# # File paths (Make sure these CSV files exist at the specified location)
# file1 = 'Astar.csv'
# file2 = 'D.csv'
# file3 = 'RRT.csv'

# try:
#     # Read the data from CSV files
#     algo1 = pd.read_csv(file1)
#     algo2 = pd.read_csv(file2)
#     algo3 = pd.read_csv(file3)
    
# except FileNotFoundError as e:
#     print(f"Error: {e}")
#     exit()

# # Add a column to identify the algorithm
# algo1['Algorithm'] = 'A* Algorithm'
# algo2['Algorithm'] = 'Dijkstra Algorithm'
# algo3['Algorithm'] = 'RRT Algorithm'

# # Combine the data into one DataFrame
# data = pd.concat([algo1, algo2, algo3])

# # Check if 'Density' and 'Execution Time' columns exist
# print(data.columns)

# # Perform ANOVA for Execution Time vs. Obstacle Density
# execution_time_anova = stats.f_oneway(
#     data[data['Density'] < 0.4]['Execution Time'],  # Group 1: Low Density
#     data[(data['Density'] >= 0.4) & (data['Density'] < 0.5)]['Execution Time'],  # Group 2: Medium Density
#     data[data['Density'] >= 0.5]['Execution Time']  # Group 3: High Density
# )

# # Perform ANOVA for Execution Time vs. Algorithm
# execution_time_algorithm_anova = stats.f_oneway(
#     data[data['Algorithm'] == 'A* Algorithm']['Execution Time'],
#     data[data['Algorithm'] == 'Dijkstra Algorithm']['Execution Time'],
#     data[data['Algorithm'] == 'RRT Algorithm']['Execution Time']
# )

# # Perform ANOVA for Path Length vs. Obstacle Density
# path_length_anova = stats.f_oneway(
#     data[data['Density'] < 0.4]['Path Length'],
#     data[(data['Density'] >= 0.4) & (data['Density'] < 0.5)]['Path Length'],
#     data[data['Density'] >= 0.5]['Path Length']
# )

# # Perform ANOVA for Path Length vs. Algorithm
# path_length_algorithm_anova = stats.f_oneway(
#     data[data['Algorithm'] == 'A* Algorithm']['Path Length'],
#     data[data['Algorithm'] == 'Dijkstra Algorithm']['Path Length'],
#     data[data['Algorithm'] == 'RRT Algorithm']['Path Length']
# )

# # Output ANOVA Results
# print("ANOVA for Execution Time vs. Obstacle Density: F-value = {:.3f}, p-value = {:.3f}".format(execution_time_anova.statistic, execution_time_anova.pvalue))
# print("ANOVA for Execution Time vs. Algorithm: F-value = {:.3f}, p-value = {:.3f}".format(execution_time_algorithm_anova.statistic, execution_time_algorithm_anova.pvalue))
# print("ANOVA for Path Length vs. Obstacle Density: F-value = {:.3f}, p-value = {:.3f}".format(path_length_anova.statistic, path_length_anova.pvalue))
# print("ANOVA for Path Length vs. Algorithm: F-value = {:.3f}, p-value = {:.3f}".format(path_length_algorithm_anova.statistic, path_length_algorithm_anova.pvalue))


# import pandas as pd
# import statsmodels.api as sm

# # File paths (Ensure these files exist in the specified location)
# file1 = 'Astar.csv'
# file2 = 'D.csv'
# file3 = 'RRT.csv'

# try:
#     # Read the data from CSV files
#     algo1 = pd.read_csv(file1)
#     algo2 = pd.read_csv(file2)
#     algo3 = pd.read_csv(file3)
    
# except FileNotFoundError as e:
#     print(f"Error: {e}")
#     exit()

# # Add a column to identify the algorithm
# algo1['Algorithm'] = 'A* Algorithm'
# algo2['Algorithm'] = 'Dijkstra Algorithm'
# algo3['Algorithm'] = 'RRT Algorithm'

# # Combine the data into one DataFrame
# data = pd.concat([algo1, algo2, algo3])

# # Check the columns and first few rows of the data to ensure correctness
# print(data.columns)
# print(data.head())

# # Define the predictors (X) and target variable (Y) for regression
# # Y = Execution Time, X₁ = Obstacle Density, X₂ = Path Complexity
# X = data[['Density', 'Visited']]
# Y = data['Execution Time']

# # Add a constant to the predictors (for the intercept term)
# X = sm.add_constant(X)

# # Fit the multiple linear regression model
# model = sm.OLS(Y, X).fit()

# # Output the regression summary (includes t-tests for coefficients)
# print(model.summary())

# # Hypothesis Testing for Regression Coefficients
# # Null Hypothesis (H₀): Coefficient = 0
# # Alternative Hypothesis (H₁): Coefficient ≠ 0
# # P-values in the summary are used to evaluate the null hypothesis

# # Prediction using the regression model
# # Define new input data for prediction (example values)
# new_data = pd.DataFrame({
#     'const': 1,  # Add the constant term
#     'Density': [0.3, 0.6],  # Example obstacle densities
#     'Visited': [1.2, 2.5]  # Example path complexities
# })

# # Predict execution time
# predictions = model.predict(new_data)

# # Output the predictions
# for i, pred in enumerate(predictions):
#     print(f"Prediction {i+1}: Execution Time = {pred:.3f} (Density={new_data.iloc[i]['Density']}, Visited={new_data.iloc[i]['Visited']})")


# import pandas as pd
# import statsmodels.api as sm
# from statsmodels.stats.outliers_influence import variance_inflation_factor
# import numpy as np

# # File names
# data_files = {'Astar.csv': 'A* Algorithm', 'D.csv': 'Dijkstra Algorithm', 'RRT.csv': 'RRT Algorithm'}
# data_frames = []

# # Read and combine data
# for file, algo in data_files.items():
#     try:
#         df = pd.read_csv(file)
#         df['Algorithm'] = algo
#         data_frames.append(df)
#     except FileNotFoundError as e:
#         print(f"Error: {e}")
#         exit()

# # Combine all dataframes
# data = pd.concat(data_frames, ignore_index=True)

# # Ensure column names match expectations
# expected_columns = ['Execution Time', 'Density', 'Visited', 'Algorithm']
# if not all(col in data.columns for col in expected_columns):
#     raise ValueError(f"Dataset columns must include: {expected_columns}")

# # Select independent and dependent variables
# X = data[['Density', 'Visited']]
# Y = data['Execution Time']

# # Add constant for intercept
# X = sm.add_constant(X)

# # Fit the regression model
# model = sm.OLS(Y, X).fit()

# # Display regression summary
# print("Regression Model Summary:")
# print(model.summary())

# # Hypothesis Testing (t-tests for coefficients)
# print("\nHypothesis Testing:")
# t_values = model.tvalues
# p_values = model.pvalues
# for i, coef in enumerate(model.params.index):
#     print(f"Coefficient: {coef}, t-value: {t_values[i]:.3f}, p-value: {p_values[i]:.3f}")

# # Model Evaluation
# r_squared = model.rsquared
# sse = np.sum(model.resid ** 2)
# print(f"\nModel Performance:")
# print(f"R-squared: {r_squared:.3f}")
# print(f"Sum of Squares Error (SSE): {sse:.3f}")

# # Predicting Execution Time for sample values
# sample_inputs = pd.DataFrame({
#     'const': 1,  # For intercept
#     'Obstacle_Density': [30, 50],
#     'Path_Complexity': [10, 20]
# })
# predictions = model.predict(sample_inputs)
# print("\nPredictions:")
# print(predictions)
