"""
Convert US Kansas dataset from 10-minute intervals to 15-minute intervals
to match SL dataset format using built-in Python libraries
"""

import csv
from datetime import datetime, timedelta

def linear_interpolate(y1, y2, x1, x2, x):
    """Linear interpolation between two points"""
    if x2 == x1:
        return y1
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1)

def convert_to_15min(input_file, output_file):
    """
    Convert CSV file from 10-minute to 15-minute intervals
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
    """
    print(f"\nProcessing {input_file}...")
    
    # Read the CSV file
    rows = []
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    
    print(f"  Original rows: {len(rows)}")
    if rows:
        print(f"  Date range: {rows[0]['date']} to {rows[-1]['date']}")
    
    # Parse dates and convert to datetime objects
    data_points = []
    for row in rows:
        dt = datetime.strptime(row['date'], '%Y-%m-%d %H:%M:%S')
        data_points.append({
            'datetime': dt,
            'dayofyear': int(row['dayofyear']),
            'timeofday': int(row['timeofday']),
            'temp': float(row['temp']),
            'dew': float(row['dew']),
            'humidity': float(row['humidity']),
            'winddir': float(row['winddir']),
            'windspeed': float(row['windspeed']),
            'pressure': float(row['pressure']),
            'cloudcover': float(row['cloudcover']),
            'solar': float(row['Solar Power Output'])
        })
    
    # Generate 15-minute interval timestamps
    if not data_points:
        print("  ⚠ No data found!")
        return
    
    start_time = data_points[0]['datetime']
    end_time = data_points[-1]['datetime']
    
    # Round start_time to nearest 15-minute mark (downward)
    start_minute = start_time.minute
    start_minute_rounded = (start_minute // 15) * 15
    start_time_rounded = start_time.replace(minute=start_minute_rounded, second=0, microsecond=0)
    
    # Generate 15-minute interval timestamps
    target_times = []
    current_time = start_time_rounded
    while current_time <= end_time:
        target_times.append(current_time)
        current_time += timedelta(minutes=15)
    
    print(f"  Target 15-minute intervals: {len(target_times)}")
    
    # Interpolate values for each target timestamp
    converted_data = []
    data_idx = 0
    
    for target_time in target_times:
        # Find the two data points that bracket this target time
        while data_idx < len(data_points) - 1 and data_points[data_idx + 1]['datetime'] < target_time:
            data_idx += 1
        
        if data_idx >= len(data_points) - 1:
            # Use last data point
            point = data_points[-1].copy()
            point['datetime'] = target_time
        elif data_points[data_idx]['datetime'] == target_time:
            # Exact match
            point = data_points[data_idx].copy()
        else:
            # Interpolate between data_points[data_idx] and data_points[data_idx + 1]
            p1 = data_points[data_idx]
            p2 = data_points[data_idx + 1]
            
            t1 = p1['datetime']
            t2 = p2['datetime']
            
            # Calculate interpolation factor
            if t2 == t1:
                factor = 0
            else:
                factor = (target_time - t1).total_seconds() / (t2 - t1).total_seconds()
            
            point = {
                'datetime': target_time,
                'dayofyear': int(p1['dayofyear'] + (p2['dayofyear'] - p1['dayofyear']) * factor),
                'timeofday': int(p1['timeofday'] + (p2['timeofday'] - p1['timeofday']) * factor),
                'temp': round(p1['temp'] + (p2['temp'] - p1['temp']) * factor, 2),
                'dew': round(p1['dew'] + (p2['dew'] - p1['dew']) * factor, 2),
                'humidity': round(p1['humidity'] + (p2['humidity'] - p1['humidity']) * factor, 2),
                'winddir': round(p1['winddir'] + (p2['winddir'] - p1['winddir']) * factor, 2),
                'windspeed': round(p1['windspeed'] + (p2['windspeed'] - p1['windspeed']) * factor, 2),
                'pressure': round(p1['pressure'] + (p2['pressure'] - p1['pressure']) * factor, 2),
                'cloudcover': round(p1['cloudcover'] + (p2['cloudcover'] - p1['cloudcover']) * factor, 2),
                'solar': round(p1['solar'] + (p2['solar'] - p1['solar']) * factor, 2)
            }
        
        # Recalculate dayofyear and timeofday from datetime
        point['dayofyear'] = target_time.timetuple().tm_yday
        point['timeofday'] = target_time.hour * 3600 + target_time.minute * 60 + target_time.second
        
        converted_data.append(point)
    
    print(f"  Converted rows: {len(converted_data)}")
    if converted_data:
        print(f"  Date range: {converted_data[0]['datetime'].strftime('%Y-%m-%d %H:%M:%S')} to {converted_data[-1]['datetime'].strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Show sample
    print(f"\n  Sample converted data (first 6 rows):")
    for i, point in enumerate(converted_data[:6]):
        print(f"    {point['datetime'].strftime('%Y-%m-%d %H:%M:%S')}, {point['dayofyear']}, {point['timeofday']}, "
              f"{point['temp']}, {point['dew']}, {point['humidity']}, {point['winddir']}, "
              f"{point['windspeed']}, {point['pressure']}, {point['cloudcover']}, {point['solar']}")
    
    # Write to CSV
    fieldnames = ['date', 'dayofyear', 'timeofday', 'temp', 'dew', 'humidity', 
                  'winddir', 'windspeed', 'pressure', 'cloudcover', 'Solar Power Output']
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for point in converted_data:
            writer.writerow({
                'date': point['datetime'].strftime('%Y-%m-%d %H:%M:%S'),
                'dayofyear': point['dayofyear'],
                'timeofday': point['timeofday'],
                'temp': point['temp'],
                'dew': point['dew'],
                'humidity': point['humidity'],
                'winddir': point['winddir'],
                'windspeed': point['windspeed'],
                'pressure': point['pressure'],
                'cloudcover': point['cloudcover'],
                'Solar Power Output': point['solar']
            })
    
    print(f"\n  [OK] Saved to {output_file}")
    
    return converted_data


if __name__ == '__main__':
    import os
    import shutil
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Files to convert
    files_to_convert = ['train.csv', 'val.csv', 'test.csv']
    
    print("="*70)
    print("Converting US Kansas dataset from 10-minute to 15-minute intervals")
    print("Using linear interpolation")
    print("="*70)
    
    for filename in files_to_convert:
        input_path = os.path.join(script_dir, filename)
        output_path = os.path.join(script_dir, filename)
        
        if os.path.exists(input_path):
            # Create backup
            backup_path = os.path.join(script_dir, f"{filename}.backup")
            if not os.path.exists(backup_path):
                shutil.copy2(input_path, backup_path)
                print(f"\nCreated backup: {backup_path}")
            
            # Convert
            convert_to_15min(input_path, output_path)
        else:
            print(f"\n⚠ Warning: {input_path} not found, skipping...")
    
    print("\n" + "="*70)
    print("Conversion complete!")
    print("="*70)
    print("\nOriginal files have been backed up with .backup extension")
    print("Converted files are ready to use with 15-minute intervals")
