import csv

# Columns to keep
columns_to_keep = ['date', 'dayofyear', 'timeofday', 'Solar Power Output']

# Process each CSV file
files = ['test.csv', 'train.csv', 'val.csv']

for file in files:
    # Read the CSV file
    with open(file, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames
    
    # Get indices of columns to keep
    indices_to_keep = [i for i, col in enumerate(fieldnames) if col in columns_to_keep]
    filtered_fieldnames = [fieldnames[i] for i in indices_to_keep]
    
    # Filter rows to keep only specified columns
    filtered_rows = []
    for row in rows:
        filtered_row = {col: row[col] for col in filtered_fieldnames}
        filtered_rows.append(filtered_row)
    
    # Write back to the same file
    with open(file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=filtered_fieldnames)
        writer.writeheader()
        writer.writerows(filtered_rows)
    
    print(f"Processed {file}: Kept {len(filtered_fieldnames)} columns out of {len(fieldnames)}")
