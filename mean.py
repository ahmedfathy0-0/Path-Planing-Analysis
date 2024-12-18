import pandas as pd

# Load the Excel file
file_path = 'Astar.csv'  # Change this to your actual file path
df = pd.read_csv(file_path)

# Calculate central tendency for Execution Time and Path Optimality
execution_time_mean = df['Execution Time'].mean()
execution_time_median = df['Execution Time'].median()

path_optimality_mean = df['optimality'].mean()
path_optimality_median = df['optimality'].median()

# Calculate variability (standard deviation) for Execution Time and Path Optimality
execution_time_std = df['Execution Time'].std()
path_optimality_std = df['optimality'].std()

# Print results
print("Central Tendency:")
print(f"Execution Time Mean: {execution_time_mean}")
print(f"Execution Time Median: {execution_time_median}")
print(f"Path Optimality Mean: {path_optimality_mean}")
print(f"Path Optimality Median: {path_optimality_median}")

print("\nVariability:")
print(f"Execution Time Standard Deviation: {execution_time_std}")
print(f"Path Optimality Standard Deviation: {path_optimality_std}")
