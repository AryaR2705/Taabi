import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import missingno as msno

# Load the data
df = pd.read_csv('taabi_final.csv')

# Sample the data to include every 50th row
df = df.iloc[::50, :]

# Reset the index after sampling
df = df.reset_index(drop=True)

# Convert timestamp to datetime
df['ts'] = pd.to_datetime(df['ts'], unit='s')

# Extract hour, day of week, month, and year from timestamp
df['hour'] = df['ts'].dt.hour
df['day_of_week'] = df['ts'].dt.dayofweek
df['month'] = df['ts'].dt.month
df['year'] = df['ts'].dt.year

# Convert runtime to hours for better readability
df['runtime_hours'] = df['runtime'] / 3600

# Check for missing values
print(df.isnull().sum())


# Create a function for plotting
def plot_figure(x, y, title, xlabel, ylabel, kind='scatter'):
    plt.figure(figsize=(12, 6))
    if kind == 'scatter':
        plt.scatter(x, y, alpha=0.5)
    elif kind == 'line':
        plt.plot(x, y)
    elif kind == 'bar':
        plt.bar(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()

# 1. Vehicle Speed over Time
plot_figure(df['ts'], df['vehiclespeed'], 'Vehicle Speed over Time', 'Time', 'Speed', kind='line')

# 2. Fuel Consumption over Time
plot_figure(df['ts'], df['fuel_consumption'], 'Fuel Consumption over Time', 'Time', 'Fuel Consumption', kind='line')

# 3. Engine Load Distribution by Hour of Day
plt.figure(figsize=(12, 6))
sns.boxplot(x='hour', y='engineload', data=df)
plt.title('Engine Load Distribution by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Engine Load')
plt.show()

# 4. Average Speed by Day of Week
avg_speed_by_day = df.groupby('day_of_week')['vehiclespeed'].mean()
plot_figure(avg_speed_by_day.index, avg_speed_by_day.values, 'Average Speed by Day of Week', 'Day of Week', 'Average Speed', kind='bar')

# 5. Fuel Economy vs Time of Day
plot_figure(df['hour'], df['fuel_economy'], 'Fuel Economy vs Time of Day', 'Hour of Day', 'Fuel Economy')

# 6. Engine Temperature over Time
plot_figure(df['ts'], df['engineoiltemp'], 'Engine Oil Temperature over Time', 'Time', 'Temperature', kind='line')

# 7. Correlation Heatmap
plt.figure(figsize=(14, 12))
correlation_matrix = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Numerical Variables')
plt.tight_layout()
plt.show()

# 8. Vehicle Speed vs Engine RPM (colored by time)
plt.figure(figsize=(12, 8))
scatter = plt.scatter(df['vehiclespeed'], df['rpm'], c=df['ts'].astype(int), cmap='viridis')
plt.colorbar(scatter, label='Time')
plt.title('Vehicle Speed vs Engine RPM (colored by time)')
plt.xlabel('Vehicle Speed')
plt.ylabel('Engine RPM')
plt.show()

# 9. Fuel Consumption Rate
df['fuel_consumption_rate'] = df['fuel_consumption'].diff() / df['ts'].diff().dt.total_seconds()
plot_figure(df['ts'][1:], df['fuel_consumption_rate'][1:], 'Fuel Consumption Rate over Time', 'Time', 'Fuel Consumption Rate (per second)', kind='line')

# 10. Cumulative Distance Traveled
df['cumulative_distance'] = df['obddistance'].cumsum()
plot_figure(df['ts'], df['cumulative_distance'], 'Cumulative Distance Traveled', 'Time', 'Distance', kind='line')

# 11. Average Speed by Month
avg_speed_by_month = df.groupby('month')['vehiclespeed'].mean()
plot_figure(avg_speed_by_month.index, avg_speed_by_month.values, 'Average Speed by Month', 'Month', 'Average Speed', kind='bar')

# 12. Average Fuel Consumption by Hour of Day
avg_fuel_consumption_by_hour = df.groupby('hour')['fuel_consumption'].mean()
plot_figure(avg_fuel_consumption_by_hour.index, avg_fuel_consumption_by_hour.values, 'Average Fuel Consumption by Hour of Day', 'Hour of Day', 'Average Fuel Consumption', kind='bar')

# 13. Pair Plot of Selected Variables
sns.pairplot(df[['vehiclespeed', 'rpm', 'fuel_consumption', 'engineoiltemp']])
plt.suptitle('Pair Plot of Selected Variables', y=1.02)
plt.show()

# 14. Rolling Averages
df['rolling_avg_speed'] = df['vehiclespeed'].rolling(window=50).mean()
df['rolling_avg_fuel_consumption'] = df['fuel_consumption'].rolling(window=50).mean()
plt.figure(figsize=(12, 6))
plt.plot(df['ts'], df['rolling_avg_speed'], label='Rolling Average Speed')
plt.plot(df['ts'], df['rolling_avg_fuel_consumption'], label='Rolling Average Fuel Consumption')
plt.xlabel('Time')
plt.ylabel('Values')
plt.title('Rolling Averages')
plt.legend()
plt.tight_layout()
plt.show()

# 15. Engine Load by Selected Gear
plt.figure(figsize=(12, 6))
sns.boxplot(x='selected_gear', y='engineload', data=df)
plt.title('Engine Load by Selected Gear')
plt.xlabel('Selected Gear')
plt.ylabel('Engine Load')
plt.show()

# 16. Speed by Selected Gear
plt.figure(figsize=(12, 6))
sns.boxplot(x='selected_gear', y='vehiclespeed', data=df)
plt.title('Vehicle Speed by Selected Gear')
plt.xlabel('Selected Gear')
plt.ylabel('Vehicle Speed')
plt.show()

# 17. Acceleration over Time
df['acceleration'] = df['vehiclespeed'].diff() / df['ts'].diff().dt.total_seconds()
plot_figure(df['ts'][1:], df['acceleration'][1:], 'Acceleration over Time', 'Time', 'Acceleration (m/s^2)', kind='line')

print("Time-based analysis complete. Please check the generated plots for insights.")
