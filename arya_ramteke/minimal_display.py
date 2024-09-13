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


# 3. Average Speed by Day of Week
avg_speed_by_day = df.groupby('day_of_week')['vehiclespeed'].mean()
plot_figure(avg_speed_by_day.index, avg_speed_by_day.values, 'Average Speed by Day of Week', 'Day of Week', 'Average Speed', kind='bar')


# 4. Fuel Consumption Rate
df['fuel_consumption_rate'] = df['fuel_consumption'].diff() / df['ts'].diff().dt.total_seconds()
plot_figure(df['ts'][1:], df['fuel_consumption_rate'][1:], 'Fuel Consumption Rate over Time', 'Time', 'Fuel Consumption Rate (per second)', kind='line')

# 5. Average Speed by Month
avg_speed_by_month = df.groupby('month')['vehiclespeed'].mean()
plot_figure(avg_speed_by_month.index, avg_speed_by_month.values, 'Average Speed by Month', 'Month', 'Average Speed', kind='bar')


# 6. Engine Load by Selected Gear
plt.figure(figsize=(12, 6))
sns.boxplot(x='selected_gear', y='engineload', data=df)
plt.title('Engine Load by Selected Gear')
plt.xlabel('Selected Gear')
plt.ylabel('Engine Load')
plt.show()


print("Time-based analysis complete. Please check the generated plots for insights.")
