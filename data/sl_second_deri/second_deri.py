import pandas as pd
import numpy as np
type = 'val'
# Load the CSV file
df = pd.read_csv(f'data/sl_t/{type}.csv')

# Columns to calculate derivatives for
columns = ['temp', 'dew', 'humidity', 'winddir', 'windspeed', 'pressure', 'cloudcover']

# Calculate first, second, and third derivatives
for col in columns:
    # First derivative
    df[f'{col}_first_deriv'] = np.gradient(df[col])
    
    # Second derivative
    df[f'{col}_second_deriv'] = np.gradient(df[f'{col}_first_deriv'])
    
    # Third derivative
    # df[f'{col}_third_deriv'] = np.gradient(df[f'{col}_second_deriv'])

# Reorder columns to ensure 'Solar Power Output' is the last column
# Get the list of all columns except 'Solar Power Output'
cols = [col for col in df.columns if col != 'Solar Power Output']
# Append 'Solar Power Output' to the end
cols.append('Solar Power Output')
# Reorder the DataFrame
df = df[cols]

# Save the new DataFrame to a CSV file
df.to_csv(f'data/sl_second_deri/{type}.csv', index=False)

print("Derivatives calculated and saved to 'new_file_with_derivatives.csv'")