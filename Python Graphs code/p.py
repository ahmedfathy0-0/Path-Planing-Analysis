import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV files
try:
    file1 = 'Astar.csv'
    file2 = 'D.csv'
    file3 = 'RRT.csv'

    # Read the data
    algo1 = pd.read_csv(file1)
    algo2 = pd.read_csv(file2)
    algo3 = pd.read_csv(file3)

except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()

# Add a column to identify the algorithm
algo1['Algorithm'] = 'A* Algorithm'
algo2['Algorithm'] = 'Dijkstra Algorithm'
algo3['Algorithm'] = 'RRT Algorithm'

# Combine the data into one DataFrame
data = pd.concat([algo1, algo2, algo3])

# Define Path Complexity based on Obstacle Density
def calculate_path_complexity(density):
    """Categorizes obstacle density into Simple, Medium, or Complex."""
    if density < 0.4:
        return 'Simple'
    elif density < 0.5:
        return 'Medium'
    else:
        return 'Complex'

# Add Path Complexity column
data['Path Complexity'] = data['Density'].apply(calculate_path_complexity)

# Define colors for each algorithm
colors = {
    'A* Algorithm': 'red',
    'Dijkstra Algorithm': 'blue',
    'RRT Algorithm': 'green'
}

# Scatter Plot: Execution Time vs. Obstacle Density
plt.figure(figsize=(10, 6))
for algo, group in data.groupby('Algorithm'):
    plt.scatter(group['Density']* 100, group['Execution Time'],
                label=algo, color=colors[algo], alpha=0.7, edgecolors='k', s=80)

plt.yscale('log')  # Set x-axis to log scale

plt.title('Execution Time vs. Obstacle Density', fontsize=14)
plt.xlabel('Obstacle Density %', fontsize=12)
plt.ylabel('Execution Time (ms)', fontsize=12)
plt.legend(title='Algorithm', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# # Scatter Plot: Execution Time vs. Path Complexity
# plt.figure(figsize=(10, 6))
# for algo, group in data.groupby('Algorithm'):
#     plt.scatter(group['Path Complexity'], group['Execution Time'],
#                 label=algo, color=colors[algo], alpha=0.7, edgecolors='k', s=80)


# plt.title('Execution Time vs. Path Complexity', fontsize=14)
# plt.xlabel('Path Complexity', fontsize=12)
# plt.ylabel('Execution Time (ms)', fontsize=12)
# plt.legend(title='Algorithm', fontsize=10)
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.tight_layout()
# plt.show()
# import pandas as pd
# import matplotlib.pyplot as plt

# try:
#     file1 = 'Astar.csv'
#     file2 = 'D.csv'
#     file3 = 'RRT.csv'

#     # Read the data
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

# # Define Path Complexity based on Obstacle Density
# def calculate_path_complexity(density):
#     """Categorizes obstacle density into Simple, Medium, or Complex."""
#     if density < 0.4:
#         return 'Simple'
#     elif density < 0.5:
#         return 'Medium'
#     else:
#         return 'Complex'

# # Add Path Complexity column
# data['Path Complexity'] = data['Density'].apply(calculate_path_complexity)

# # Define colors for each algorithm
# colors = {
#     'A* Algorithm': 'red',
#     'Dijkstra Algorithm': 'blue',
#     'RRT Algorithm': 'green'
# }

# # Bar Plot: Execution Time vs. Path Complexity
# plt.figure(figsize=(10, 6))

# # Aggregate the data by Path Complexity and Algorithm, and calculate the mean Execution Time
# agg_data = data.groupby(['Path Complexity', 'Algorithm'])['Execution Time'].mean().unstack()

# # Plotting bars for each Path Complexity for every algorithm
# agg_data.plot(kind='bar', color=[colors[algo] for algo in agg_data.columns], alpha=0.7, edgecolor='black', width=0.8)

# plt.title('Execution Time vs. Path Complexity', fontsize=14)
# plt.xlabel('Path Complexity', fontsize=12)
# plt.ylabel('Average Execution Time (ms)', fontsize=12)
# plt.legend(title='Algorithm', fontsize=10)
# plt.xticks(rotation=0)
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.tight_layout()
# plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt

# try:
#     file1 = 'Astar.csv'
#     file2 = 'D.csv'
#     file3 = 'RRT.csv'

#     # Read the data
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

# # Define colors for each algorithm
# colors = {
#     'A* Algorithm': 'red',
#     'Dijkstra Algorithm': 'blue',
#     'RRT Algorithm': 'green'
# }

# # Plotting Histogram: Execution Time Distribution for each Algorithm
# plt.figure(figsize=(10, 6))

# # Plot histograms for each algorithm's execution time
# for algo, group in data.groupby('Algorithm'):
#     plt.hist(group['Execution Time'], bins=20, alpha=0.7, label=algo, color=colors[algo], edgecolor='black')
# plt.xscale('log')
# plt.title('Execution Time Distribution by Algorithm', fontsize=14)
# plt.xlabel('Execution Time (ms)', fontsize=12)
# plt.ylabel('Frequency', fontsize=12)
# plt.legend(title='Algorithm', fontsize=10)
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.tight_layout()
# plt.show()


