import pandas as pd

# Read the CSV file
df = pd.read_csv('val.csv')

# Convert datetime column to datetime type
df["date"] = pd.to_datetime(df["date"])

# Add dayofyear and timeofday columns
df["dayofyear"] = df["date"].dt.dayofyear
df["timeofday"] = df["date"].dt.hour * 3600 + df["date"].dt.minute * 60 + df["date"].dt.second

# Rearranging columns so that dayofyear and timeofday come right after datetime
df = df[["date", "dayofyear", "timeofday", "temp", "dew", "humidity", "winddir", "windspeed", "pressure", "cloudcover", "Solar Power Output"]]

# Display the DataFrame (for your reference)
print(df)

# Save the updated DataFrame back to CSV
df.to_csv('val.csv', index=False)