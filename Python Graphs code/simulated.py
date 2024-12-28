import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your datasets
data_astar = pd.read_csv("simulated astar.csv")  # Replace with your A* dataset file
data_dijkstra = pd.read_csv("simulated ijkstra result.csv")  # Replace with your Dijkstra dataset file

# Add a new column to label the algorithms
data_astar['Algorithm'] = 'A*'
data_dijkstra['Algorithm'] = 'Dijkstra'

# Combine both datasets into one
data = pd.concat([data_astar, data_dijkstra], ignore_index=True)
mean_data = data.groupby(['Population', 'Algorithm'], as_index=False)['Execution Time'].mean()

# Time Series Plot (Population vs Execution Time)
palette = {'A*': 'blue', 'Dijkstra': 'green'}

sns.lineplot(data=mean_data, x="Population", y="Execution Time", hue="Algorithm", palette=palette, markers=True)
plt.title("Mean Execution Time vs Population Size (Grouped by Algorithm)")
plt.xlabel("Population Size")
plt.ylabel("Mean Execution Time (ms)")
plt.legend(title="Algorithm")
plt.grid()
plt.show()

# Boxplot (Population vs Execution Time)
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x="Population", y="Execution Time", hue="Algorithm")
plt.title("Execution Time Variability Across Algorithms")
plt.xlabel("Population Size")
plt.ylabel("Execution Time (ms)")
plt.grid()
plt.show()

# Scatter Plot (Obstacle Density vs Execution Time)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x="Density", y="Execution Time", hue="Algorithm", style="Population", palette="Set1")
plt.title("Execution Time vs Obstacle Density (Grouped by Algorithm)")
plt.xlabel("Obstacle Density (%)")
plt.ylabel("Execution Time (ms)")
plt.legend(title="Algorithm")
plt.grid()
plt.show()

