
import pandas as pd

# Read the CSV file
df = pd.read_csv('test.csv')

df = df.iloc[:-3]

df.to_csv('test.csv', index=False)
