import requests
import pandas as pd
from datetime import datetime, timedelta

# Define the start and end time for the last 7 days
end_time = datetime.utcnow()
start_time = end_time - timedelta(days=7)

# Format the dates
start_time_str = start_time.strftime('%Y-%m-%dT%H:%M:%S')
end_time_str = end_time.strftime('%Y-%m-%dT%H:%M:%S')

# Construct the API URL
url = f"https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&starttime={start_time_str}&endtime={end_time_str}"

# Fetch the data
response = requests.get(url)
csv_data = response.content.decode('utf-8')

# Save the data to a CSV file
with open('earthquake_data.csv', 'w') as file:
    file.write(csv_data)

