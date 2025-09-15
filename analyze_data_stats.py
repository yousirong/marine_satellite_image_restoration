#!/usr/bin/env python3
import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

def analyze_csv_files(data_dir, data_type, sample_size=10):
    """Analyze CSV files in a directory and return statistics"""
    
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return None
    
    print(f"\n{'='*60}")
    print(f"ANALYZING {data_type.upper()} DATA")
    print(f"{'='*60}")
    print(f"Total files: {len(csv_files)}")
    
    # Sample files for detailed analysis
    sample_files = csv_files[:sample_size]
    all_values = []
    file_stats = []
    
    for i, csv_file in enumerate(sample_files):
        try:
            data = np.loadtxt(csv_file, delimiter=',')
            all_values.extend(data.flatten())
            
            # File-level statistics
            stats = {
                'file': os.path.basename(csv_file),
                'shape': data.shape,
                'min': np.min(data),
                'max': np.max(data),
                'mean': np.mean(data),
                'std': np.std(data),
                'unique_values': len(np.unique(data)),
                'zeros': np.sum(data == 0),
                'negatives': np.sum(data < 0),
                'nan_inf': np.sum(~np.isfinite(data))
            }
            file_stats.append(stats)
            
            if i < 3:  # Show first 3 files in detail
                print(f"\nFile {i+1}: {os.path.basename(csv_file)}")
                print(f"  Shape: {data.shape}")
                print(f"  Range: [{np.min(data):.6e}, {np.max(data):.6e}]")
                print(f"  Mean: {np.mean(data):.6e}, Std: {np.std(data):.6e}")
                print(f"  Unique values: {len(np.unique(data))}")
                print(f"  Zeros: {np.sum(data == 0)}, Negatives: {np.sum(data < 0)}")
                print(f"  Non-finite: {np.sum(~np.isfinite(data))}")
                
                # Show some extreme values
                if len(np.unique(data)) < 20:
                    print(f"  All unique values: {sorted(np.unique(data))}")
                else:
                    print(f"  Sample values: {np.unique(data)[:10]}")
        
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            continue
    
    # Overall statistics
    if all_values:
        all_values = np.array(all_values)
        print(f"\n{'='*40}")
        print(f"OVERALL {data_type.upper()} STATISTICS")
        print(f"{'='*40}")
        print(f"Total pixels analyzed: {len(all_values)}")
        print(f"Global range: [{np.min(all_values):.6e}, {np.max(all_values):.6e}]")
        print(f"Global mean: {np.mean(all_values):.6e}")
        print(f"Global std: {np.std(all_values):.6e}")
        print(f"Global median: {np.median(all_values):.6e}")
        
        # Percentiles
        percentiles = [0.1, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]
        perc_values = np.percentile(all_values, percentiles)
        print(f"\nPercentiles:")
        for p, v in zip(percentiles, perc_values):
            print(f"  {p:5.1f}%: {v:.6e}")
        
        # Special value analysis
        zeros = np.sum(all_values == 0)
        negatives = np.sum(all_values < 0)
        positives = np.sum(all_values > 0)
        large_values = np.sum(np.abs(all_values) > 1000)
        extreme_values = np.sum(np.abs(all_values) > 1e10)
        
        print(f"\nSpecial Values Analysis:")
        print(f"  Zeros: {zeros} ({zeros/len(all_values)*100:.2f}%)")
        print(f"  Negatives: {negatives} ({negatives/len(all_values)*100:.2f}%)")
        print(f"  Positives: {positives} ({positives/len(all_values)*100:.2f}%)")
        print(f"  |value| > 1000: {large_values} ({large_values/len(all_values)*100:.2f}%)")
        print(f"  |value| > 1e10: {extreme_values} ({extreme_values/len(all_values)*100:.2f}%)")
        
        # Check for specific patterns
        if data_type.lower() == 'mask':
            unique_vals = np.unique(all_values)
            print(f"  Unique values in mask: {unique_vals}")
        elif data_type.lower() == 'gt':
            thousand_values = np.sum(all_values == 1000)
            print(f"  Values == 1000: {thousand_values} ({thousand_values/len(all_values)*100:.2f}%)")
        
        return {
            'data_type': data_type,
            'total_files': len(csv_files),
            'sample_size': len(sample_files),
            'total_pixels': len(all_values),
            'values': all_values,
            'file_stats': file_stats
        }
    
    return None

def main():
    base_dir = "/home/juneyonglee/Desktop/AY_ust/myhdd/GOCI_RRS/daily_results/band3/2021/20210101"
    
    # First time directory (001641)
    time_dir = os.path.join(base_dir, "001641", "degree")
    
    data_types = ['gt', 'recon', 'mask']
    results = {}
    
    for data_type in data_types:
        data_dir = os.path.join(time_dir, data_type)
        if os.path.exists(data_dir):
            results[data_type] = analyze_csv_files(data_dir, data_type, sample_size=15)
        else:
            print(f"Directory not found: {data_dir}")
    
    # Compare across data types
    print(f"\n{'='*60}")
    print("COMPARISON ACROSS DATA TYPES")
    print(f"{'='*60}")
    
    for data_type in data_types:
        if data_type in results and results[data_type]:
            result = results[data_type]
            values = result['values']
            print(f"\n{data_type.upper()}:")
            print(f"  Range: [{np.min(values):.6e}, {np.max(values):.6e}]")
            print(f"  Mean ± Std: {np.mean(values):.6e} ± {np.std(values):.6e}")
            print(f"  Extreme values (|x| > 1e10): {np.sum(np.abs(values) > 1e10)} / {len(values)}")

if __name__ == "__main__":
    main()