import os
import re

# Define the coordinates for the 512x512 patches (Nakdong and Saemangeum)
nakdong_x_min, nakdong_x_max = 2208, 2720
nakdong_y_min, nakdong_y_max = 2925, 3437

saemangeum_x_min, saemangeum_x_max = 1723, 2235
saemangeum_y_min, saemangeum_y_max = 2511, 3023

# Function to extract row and column from the filename using regex
def extract_row_col(filename):
    match = re.search(r'r(\d+)_c(\d+)', filename)
    if match:
        row = int(match.group(1))
        col = int(match.group(2))
        return row, col
    return None, None

# Function to determine if a patch is within the Nakdong or Saemangeum 512x512 patch
def determine_region_from_patch(row, col):
    # Check if the coordinates fall within Nakdong's 512x512 patch
    if nakdong_x_min <= row < nakdong_x_max and nakdong_y_min <= col < nakdong_y_max:
        return 'nak'
    # Check if the coordinates fall within Saemangeum's 512x512 patch
    elif saemangeum_x_min <= row < saemangeum_x_max and saemangeum_y_min <= col < saemangeum_y_max:
        return 'sae'
    # Return None if it doesn't belong to either region
    return None

# Function to rename the file based on the determined region (nak or sae)
def rename_file(file_path, new_region):
    directory, filename = os.path.split(file_path)
    # Check if the file already contains 'nak' or 'sae'
    if 'nak' in filename or 'sae' in filename:
        new_filename = re.sub(r'(nak|sae)', new_region, filename)
    else:
        # If neither 'nak' nor 'sae' is in the filename, append the correct region
        new_filename = new_region + "_" + filename
    new_file_path = os.path.join(directory, new_filename)
    if new_file_path != file_path:
        print(f"Renaming {file_path} to {new_file_path}")
        os.rename(file_path, new_file_path)

# Function to process all files and update the file names based on their region
def process_directory(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.tiff'):
                file_path = os.path.join(root, file)
                # Extract row and col from the filename
                row, col = extract_row_col(file)
                if row is not None and col is not None:
                    # Determine the correct region based on the row and column
                    correct_region = determine_region_from_patch(row, col)
                    if correct_region:
                        rename_file(file_path, correct_region)

# Function to process multiple bands and NaN folders
def process_all_bands(base_dir):
    bands = ['band_2', 'band_3', 'band_4']  # List of band folders to process
    for band in bands:
        # Define the train and test directories for each band
        train_dir = os.path.join(base_dir, band, 'train')
        test_dir = os.path.join(base_dir, band, 'test')
        
        # Process the train and test directories for each band
        print(f"Processing {band} train directory...")
        process_directory(train_dir)
        print(f"Processing {band} test directory...")
        process_directory(test_dir)

# Example usage
base_dir = '/media/juneyonglee/My Book/Preprocessed/GOCI/L2_Rrs_new'

process_all_bands(base_dir)
