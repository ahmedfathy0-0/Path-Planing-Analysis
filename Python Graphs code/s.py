import pandas as pd
import matplotlib.pyplot as plt

# Assuming your data is in a CSV file named 'vehicle_maintenance_data.csv'
data = pd.read_csv('vehicle_maintenance_data.csv')

# Filter data for electric and diesel vehicles
electric_data = data[data['Fuel_Type'] == 'Electric']
diesel_data = data[data['Fuel_Type'] == 'Diesel']
electric_data = electric_data.head(50)
diesel_data = diesel_data.head(50)

# Create the box plots
plt.figure(figsize=(10, 6))

# Electric box plot with blue color
plt.boxplot([electric_data['Service_History']], 
            positions=[1], widths=0.5, patch_artist=True, 
            boxprops=dict(facecolor='lightblue', color='black'),
            flierprops=dict(markerfacecolor='blue', marker='o', markersize=7, linestyle='none'),
            medianprops=dict(color='blue', linewidth=2))

# Diesel box plot with green color
plt.boxplot([diesel_data['Service_History']], 
            positions=[2], widths=0.5, patch_artist=True, 
            boxprops=dict(facecolor='lightgreen', color='black'),
            flierprops=dict(markerfacecolor='green', marker='o', markersize=7, linestyle='none'),
            medianprops=dict(color='green', linewidth=2))

# Add labels and title
plt.xticks([1, 2], ['Electric', 'Diesel'])
plt.xlabel('Fuel Type')
plt.ylabel('Service History')
plt.title('Comparison of Maintenance Frequency (Service History) for Electric and Diesel Vehicles')

# Show the plot
plt.show()