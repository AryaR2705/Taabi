import matplotlib
matplotlib.use('Agg')
from flask import Flask, render_template, request, jsonify
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
import base64

app = Flask(__name__)

df = pd.read_csv('taabi_final.csv')
df = df.iloc[::50, :].reset_index(drop=True)
df['ts'] = pd.to_datetime(df['ts'], unit='s')
df['hour'] = df['ts'].dt.hour
df['day_of_week'] = df['ts'].dt.dayofweek
df['month'] = df['ts'].dt.month
df['year'] = df['ts'].dt.year
df['runtime_hours'] = df['runtime'] / 3600

def create_plot(kind, x, y, title, xlabel, ylabel):
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
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    return image_base64

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    plot_type = request.form.get('plot_type')
    if plot_type == 'speed_over_time':
        img_base64 = create_plot('line', df['ts'], df['vehiclespeed'], 'Vehicle Speed over Time', 'Time', 'Speed')
    elif plot_type == 'fuel_consumption_over_time':
        img_base64 = create_plot('line', df['ts'], df['fuel_consumption'], 'Fuel Consumption over Time', 'Time', 'Fuel Consumption')
    elif plot_type == 'engine_load_distribution':
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='hour', y='engineload', data=df)
        plt.title('Engine Load Distribution by Hour of Day')
        plt.xlabel('Hour of Day')
        plt.ylabel('Engine Load')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
    elif plot_type == 'avg_speed_by_day':
        avg_speed_by_day = df.groupby('day_of_week')['vehiclespeed'].mean()
        img_base64 = create_plot('bar', avg_speed_by_day.index, avg_speed_by_day.values, 'Average Speed by Day of Week', 'Day of Week', 'Average Speed')
    elif plot_type == 'fuel_economy_vs_time':
        img_base64 = create_plot('line', df['hour'], df['fuel_economy'], 'Fuel Economy vs Time of Day', 'Hour of Day', 'Fuel Economy')
    elif plot_type == 'engine_temp_over_time':
        img_base64 = create_plot('line', df['ts'], df['engineoiltemp'], 'Engine Oil Temperature over Time', 'Time', 'Temperature')
    elif plot_type == 'correlation_heatmap':
        plt.figure(figsize=(14, 12))
        correlation_matrix = df.select_dtypes(include=[np.number]).corr()
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Heatmap of Numerical Variables')
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
    elif plot_type == 'speed_vs_rpm':
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(df['vehiclespeed'], df['rpm'], c=df['ts'].astype(int), cmap='viridis')
        plt.colorbar(scatter, label='Time')
        plt.title('Vehicle Speed vs Engine RPM (colored by time)')
        plt.xlabel('Vehicle Speed')
        plt.ylabel('Engine RPM')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
    elif plot_type == 'fuel_consumption_rate':
        df['fuel_consumption_rate'] = df['fuel_consumption'].diff() / df['ts'].diff().dt.total_seconds()
        img_base64 = create_plot('line', df['ts'][1:], df['fuel_consumption_rate'][1:], 'Fuel Consumption Rate over Time', 'Time', 'Fuel Consumption Rate (per second)')
    elif plot_type == 'cumulative_distance':
        df['cumulative_distance'] = df['obddistance'].cumsum()
        img_base64 = create_plot('line', df['ts'], df['cumulative_distance'], 'Cumulative Distance Traveled', 'Time', 'Distance')
    elif plot_type == 'avg_speed_by_month':
        avg_speed_by_month = df.groupby('month')['vehiclespeed'].mean()
        img_base64 = create_plot('bar', avg_speed_by_month.index, avg_speed_by_month.values, 'Average Speed by Month', 'Month', 'Average Speed')
    elif plot_type == 'avg_fuel_consumption_by_hour':
        avg_fuel_consumption_by_hour = df.groupby('hour')['fuel_consumption'].mean()
        img_base64 = create_plot('bar', avg_fuel_consumption_by_hour.index, avg_fuel_consumption_by_hour.values, 'Average Fuel Consumption by Hour of Day', 'Hour of Day', 'Average Fuel Consumption')
    elif plot_type == 'pair_plot':
        sns.pairplot(df[['vehiclespeed', 'rpm', 'fuel_consumption', 'engineoiltemp']])
        plt.suptitle('Pair Plot of Selected Variables', y=1.02)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
    elif plot_type == 'rolling_averages':
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
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
    elif plot_type == 'engine_load_by_gear':
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='selected_gear', y='engineload', data=df)
        plt.title('Engine Load by Selected Gear')
        plt.xlabel('Selected Gear')
        plt.ylabel('Engine Load')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
    elif plot_type == 'speed_by_gear':
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='selected_gear', y='vehiclespeed', data=df)
        plt.title('Vehicle Speed by Selected Gear')
        plt.xlabel('Selected Gear')
        plt.ylabel('Vehicle Speed')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
    elif plot_type == 'acceleration_over_time':
        df['acceleration'] = df['vehiclespeed'].diff() / df['ts'].diff().dt.total_seconds()
        img_base64 = create_plot('line', df['ts'][1:], df['acceleration'][1:], 'Acceleration over Time', 'Time', 'Acceleration (m/s^2)')
    else:
        img_base64 = None

    return jsonify({'image': img_base64})

@app.route('/vehicle_map')
def vehicle_map():
    return render_template('vehicle_map_with_timestamps.html')

if __name__ == '__main__':
    app.run(debug=True)
