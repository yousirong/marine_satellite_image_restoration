#!/usr/bin/env python3
import os
import glob
import numpy as np

def quick_analyze_directory(data_dir, data_type):
    """Quick analysis of all CSV files in directory"""
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    if not csv_files:
        return None
    
    # Sample a few files
    sample_files = csv_files[:5]
    all_values = []
    
    for csv_file in sample_files:
        try:
            data = np.loadtxt(csv_file, delimiter=',')
            all_values.extend(data.flatten())
        except Exception as e:
            continue
    
    if all_values:
        all_values = np.array(all_values)
        return {
            'total_files': len(csv_files),
            'sample_pixels': len(all_values),
            'min': np.min(all_values),
            'max': np.max(all_values),
            'mean': np.mean(all_values),
            'std': np.std(all_values),
            'zeros': np.sum(all_values == 0),
            'extremes': np.sum(np.abs(all_values) > 1e10)
        }
    return None

def main():
    base_dir = "/home/juneyonglee/Desktop/AY_ust/myhdd/GOCI_RRS/daily_results/band3/2021/20210101"
    
    # Check different time directories
    time_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    time_dirs = sorted(time_dirs)
    
    print(f"Found time directories: {time_dirs}")
    print(f"\n{'='*80}")
    print("CROSS-TIME COMPARISON")
    print(f"{'='*80}")
    
    for time_dir in time_dirs[:4]:  # First 4 time directories
        time_path = os.path.join(base_dir, time_dir, "degree")
        print(f"\nTime: {time_dir}")
        print("-" * 40)
        
        for data_type in ['gt', 'recon', 'mask']:
            data_dir = os.path.join(time_path, data_type)
            if os.path.exists(data_dir):
                result = quick_analyze_directory(data_dir, data_type)
                if result:
                    print(f"{data_type.upper():>6}: Range=[{result['min']:.2e}, {result['max']:.2e}], "
                          f"Mean={result['mean']:.2e}, Extremes={result['extremes']}/{result['sample_pixels']}")
                else:
                    print(f"{data_type.upper():>6}: No data")
    
    # Check if this is a pattern across different dates
    print(f"\n{'='*80}")
    print("CHECKING OTHER DATES")
    print(f"{'='*80}")
    
    dates_dir = "/home/juneyonglee/Desktop/AY_ust/myhdd/GOCI_RRS/daily_results/band3/2021"
    available_dates = [d for d in os.listdir(dates_dir) if d.startswith('2021') and os.path.isdir(os.path.join(dates_dir, d))]
    
    for date in available_dates[:3]:  # Check first 3 dates
        date_path = os.path.join(dates_dir, date)
        first_time = sorted(os.listdir(date_path))[0]  # First time of day
        
        print(f"\nDate: {date}, Time: {first_time}")
        print("-" * 40)
        
        for data_type in ['gt', 'recon', 'mask']:
            data_dir = os.path.join(date_path, first_time, "degree", data_type)
            if os.path.exists(data_dir):
                result = quick_analyze_directory(data_dir, data_type)
                if result:
                    print(f"{data_type.upper():>6}: Range=[{result['min']:.2e}, {result['max']:.2e}], "
                          f"Mean={result['mean']:.2e}, Extremes={result['extremes']}/{result['sample_pixels']}")
                else:
                    print(f"{data_type.upper():>6}: No data")

if __name__ == "__main__":
    main()