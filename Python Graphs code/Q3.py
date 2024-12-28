import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Load the data for each algorithm (replace file paths with your actual paths)
# Assuming the data has two columns: 'density' and 'time'
data_a_star = pd.read_csv('Astar.csv')  # A* algorithm data
data_rrt = pd.read_csv('RRT.csv')  # RRT algorithm data
data_dijkstra = pd.read_csv('D.csv')  # Dijkstra algorithm data

# Function to perform regression and hypothesis testing
def perform_regression_analysis(data, algorithm_name):
    X = sm.add_constant(data['Density'])  # Add constant (intercept) to the density column
    Y = data['Execution Time']  # Execution time column
    
    # Fit the regression model
    model = sm.OLS(Y, X)
    results = model.fit()
    
    # Print the regression summary
    print(f"Regression results for {algorithm_name}:")
    print(results.summary())
    
    # Hypothesis testing for coefficients
    t_stat_x = results.tvalues[1]  # t-statistic for obstacle density coefficient
    p_value_x = results.pvalues[1]  # p-value for obstacle density coefficient
    
    # Null and Alternative Hypotheses
    print("\nHypothesis Testing for the Regression Coefficient of Obstacle Density:")
    print("Null Hypothesis (H₀): The regression coefficient for obstacle density is zero (β = 0).")
    print("Alternative Hypothesis (H₁): The regression coefficient for obstacle density is non-zero (β ≠ 0).")
    
    # Perform the t-test and decision rule based on the p-value
    print(f"t-statistic for obstacle density coefficient (β): {t_stat_x}")
    print(f"p-value for obstacle density coefficient (β): {p_value_x}")
    
    if p_value_x < 0.05:
        print("Since the p-value is less than 0.05, we reject the null hypothesis H₀.")
        print("This indicates that the regression coefficient for obstacle density is statistically significant.")
    else:
        print("Since the p-value is greater than 0.05, we fail to reject the null hypothesis H₀.")
        print("This suggests that the regression coefficient for obstacle density is not statistically significant.")
    
    # Predict execution time using the fitted model
    predicted_time = results.predict(X)
    
    # Calculate residuals and SSE (Sum of Squared Errors)
    residuals = Y - predicted_time
    SSE = np.sum(residuals**2)
    
    # Print residuals and SSE
    print(f"Sum of Squared Errors (SSE) for {algorithm_name}: {SSE}")
    
    # Return the results for further analysis
    return results, predicted_time, residuals

# Perform regression analysis for each algorithm
results_a_star, predicted_a_star, residuals_a_star = perform_regression_analysis(data_a_star, 'A*')
results_rrt, predicted_rrt, residuals_rrt = perform_regression_analysis(data_rrt, 'RRT')
results_dijkstra, predicted_dijkstra, residuals_dijkstra = perform_regression_analysis(data_dijkstra, 'Dijkstra')

# Plot the regression results for all three algorithms
plt.figure(figsize=(12, 8))

# A* Algorithm Plot
plt.subplot(3, 1, 1)
plt.scatter(data_a_star['Density'], data_a_star['Execution Time'], label='Actual Data (A*)', color='blue')
plt.plot(data_a_star['Density'], predicted_a_star, label='Fitted Line (A*)', color='red')
plt.title("A* Algorithm - Regression")
plt.xlabel('Obstacle Density')
plt.ylabel('Execution Time')
plt.legend()

# RRT Algorithm Plot
plt.subplot(3, 1, 2)
plt.scatter(data_rrt['Density'], data_rrt['Execution Time'], label='Actual Data (RRT)', color='green')
plt.plot(data_rrt['Density'], predicted_rrt, label='Fitted Line (RRT)', color='red')
plt.title("RRT Algorithm - Regression")
plt.xlabel('Obstacle Density')
plt.ylabel('Execution Time')
plt.legend()

# Dijkstra Algorithm Plot
plt.subplot(3, 1, 3)
plt.scatter(data_dijkstra['Density'], data_dijkstra['Execution Time'], label='Actual Data (Dijkstra)', color='purple')
plt.plot(data_dijkstra['Density'], predicted_dijkstra, label='Fitted Line (Dijkstra)', color='red')
plt.title("Dijkstra Algorithm - Regression")
plt.xlabel('Obstacle Density')
plt.ylabel('Execution Time')
plt.legend()

plt.tight_layout()
plt.show()
