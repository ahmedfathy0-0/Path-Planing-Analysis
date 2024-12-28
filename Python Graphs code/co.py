# # import pandas as pd
# # import matplotlib.pyplot as plt
# # import seaborn as sns

# # # Load the CSV files
# # try:
# #     file1 = 'Astar.csv'  
# #     file2 = 'D.csv'
# #     file3 = 'RRT.csv'

# #     # Read the data
# #     algo1 = pd.read_csv(file1)
# #     algo2 = pd.read_csv(file2)
# #     algo3 = pd.read_csv(file3)

# # except FileNotFoundError as e:
# #     print(f"Error: {e}")
# #     exit()

# # # Add a column to identify the algorithm
# # algo1['Algorithm'] = 'A* Algorithm'
# # algo2['Algorithm'] = 'Dijkstra Algorithm'
# # algo3['Algorithm'] = 'RRT Algorithm'

# # # Combine the data into one DataFrame
# # data = pd.concat([algo1, algo2, algo3])

# # # # Plotting Boxplot: Execution Time vs Algorithm
# # # plt.figure(figsize=(10, 6))
# # # sns.boxplot(x='Algorithm', y='Execution Time', data=data, palette="Set2")

# # # plt.title('Comparison of Execution Times Across Algorithms')
# # # plt.xlabel('Algorithm')
# # # plt.ylabel('Execution Time (ms)')
# # # plt.grid(True, linestyle='--', alpha=0.6)
# # # plt.tight_layout()
# # # plt.show()

# # # Calculate and print the correlation between Obstacle Density and Path Optimality
# # correlation = algo1[['Density', 'optimality']].corr()

# # # Display the correlation matrix
# # print("Correlation Matrix between Obstacle Density and Path Optimality:\n")
# # print(correlation)

# # # # Plotting the correlation with a scatter plot
# # # plt.figure(figsize=(10, 6))
# # # sns.scatterplot(x='Density', y='Path Optimality', data=data, hue='Algorithm', palette='Set2', alpha=0.7)

# # # plt.title('Correlation Between Obstacle Density and Path Optimality')
# # # plt.xlabel('Obstacle Density')
# # # plt.ylabel('Path Optimality')
# # # plt.grid(True, linestyle='--', alpha=0.6)
# # # plt.tight_layout()
# # # plt.show()
# import pandas as pd
# import matplotlib.pyplot as plt

# # Assuming you already have the data loaded as algo1, algo2, and algo3
# file1 = 'Astar.csv'  
# file2 = 'D.csv'
# file3 = 'RRT.csv'
# # Read the data
# algo1 = pd.read_csv(file1)
# algo2 = pd.read_csv(file2)
# algo3 = pd.read_csv(file3)

# # Add a column to identify the algorithm
# algo1['Algorithm'] = 'A* Algorithm'
# algo2['Algorithm'] = 'Dijkstra Algorithm'
# algo3['Algorithm'] = 'RRT Algorithm'

# # Combine the data into one DataFrame
# data = pd.concat([algo1, algo2, algo3])

# # Calculate the correlation for each algorithm
# correlation_algo1 = algo1[['Density', 'optimality']].corr().iloc[0, 1]  # Correlation for A* Algorithm
# correlation_algo2 = algo2[['Density', 'optimality']].corr().iloc[0, 1]  # Correlation for Dijkstra Algorithm
# correlation_algo3 = algo3[['Density', 'optimality']].corr().iloc[0, 1]  # Correlation for RRT Algorithm

# # Create a DataFrame to store the correlation values
# correlation_data = pd.DataFrame({
#     'Algorithm': ['A* Algorithm', 'Dijkstra Algorithm', 'RRT Algorithm'],
#     'Correlation between Density and Optimality': [correlation_algo1, correlation_algo2, correlation_algo3]
# })

# # Plotting the table
# fig, ax = plt.subplots(figsize=(8, 4))  # Adjust the figure size as needed
# ax.axis('off')  # Hide the axes

# # Plot the table
# table = ax.table(cellText=correlation_data.values,
#                 colLabels=correlation_data.columns,
#                 cellLoc='center', loc='center', colColours=['#f1f1f1']*2)

# # Customize table appearance (optional)
# table.auto_set_font_size(False)
# table.set_fontsize(12)
# table.scale(1.2, 1.2)

# plt.title('Correlation between Obstacle Density and Path Optimality for each Algorithm')
# plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define file paths for each algorithm
file_paths = {
    "RRT": "RRT.csv",         # Replace with the actual file path
    "A*": "Astar.csv",        # Replace with the actual file path
    "Dijkstra": "D.csv"       # Replace with the actual file path
}

# Function to calculate correlation and plot results
def calculate_correlation(file_path, algorithm):
    # Load the data from the CSV file
    df = pd.read_csv(file_path)
    
    # Ensure the file contains these columns: "Density", "Execution Time", "Visited Nodes"
    # Calculate Computational Efficiency
    df["Computational Efficiency"] = df["Visited"] / df["Execution Time"]
    
    # Calculate correlation
    correlation = np.corrcoef(df["Density"], df["Computational Efficiency"])[0, 1]
    print(f"Correlation for {algorithm}: {correlation:.2f}")
    
    # Scatter plot
    plt.scatter(df["Density"], df["Computational Efficiency"], label=f"{algorithm}")
    plt.title(f"{algorithm} - Correlation: {correlation:.2f}")
    plt.xlabel("Obstacle Density")
    plt.ylabel("Computational Efficiency")
    plt.legend()
    plt.grid()
    plt.show()

# Process each file
for algorithm, file_path in file_paths.items():
    calculate_correlation(file_path, algorithm)
