import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('taabi.csv')

# Remove columns that contain only null values
df_cleaned = df.dropna(axis=1, how='all')

# List of columns to remove
columns_to_remove = [
    'uniqueid', 'fl_level', 'can_raw_data', 'vibration_status', 
    'engine_throttle_valve2_pos', 'drivers_demand_engine_torque_percent', 
    'accelerator_pedal_pos_2', 'engine_torque_mode', 'fuel_rate', 'pluscode', 'adblue_level'
]

# Remove the specified columns
df_cleaned = df.drop(columns=columns_to_remove)

# Check remaining columns
print(df_cleaned.columns)
# Save the cleaned dataset to a CSV file
df_cleaned.to_csv('taabi_final.csv', index=False)

print("Cleaned dataset saved as 'taabi_final.csv'")
