import pandas as pd
import folium

# Load the cleaned dataset
df_cleaned = pd.read_csv('taabi_final.csv')

# Convert timestamp to datetime if necessary
df_cleaned['ts'] = pd.to_datetime(df_cleaned['ts'], unit='s')

# Get the mean of latitude and longitude to center the map
latitude_center = df_cleaned['lat'].mean()
longitude_center = df_cleaned['lng'].mean()

# Create a folium map centered at the mean latitude and longitude
map_ = folium.Map(location=[latitude_center, longitude_center], zoom_start=12)

# Iterate through the dataframe and add every 50th point to the map
for index, row in df_cleaned.iloc[::50].iterrows():  # Skip every 50 rows
    timestamp = row['ts'].strftime('%Y-%m-%d %H:%M:%S')  # Format the timestamp
    folium.Marker([row['lat'], row['lng']], 
                  popup=f"Vehicle at index {index}<br>Timestamp: {timestamp}").add_to(map_)

# Save the map to an HTML file
map_.save('vehicle_map_with_timestamps.html')

print("Map with reduced points and timestamps has been created and saved as 'vehicle_map_with_timestamps.html'")
