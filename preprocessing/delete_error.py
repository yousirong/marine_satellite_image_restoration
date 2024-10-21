import os
import re
from datetime import datetime

# Directory paths to search for files
save_path = '/media/juneyonglee/My Book/Preprocessed/GOCI/L2_Rrs_new'

# Date range to check
start_date = datetime.strptime('2011-04-01', '%Y-%m-%d')
end_date = datetime.strptime('2015-09-15', '%Y-%m-%d')

# Regex pattern to extract the date from the filename (assuming format is YYYY-MM-DD)
date_pattern = re.compile(r'_(\d{4}-\d{2}-\d{2})_')

def delete_files_in_date_range(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Check if the filename contains a date in the specified range
            match = date_pattern.search(file)
            if match:
                file_date_str = match.group(1)
                file_date = datetime.strptime(file_date_str, '%Y-%m-%d')
                # Check if the file date is within the specified range
                if start_date <= file_date <= end_date:
                    file_path = os.path.join(root, file)
                    print(f"Deleting {file_path}")
                    os.remove(file_path)

# Run the deletion process for the entire save_path directory
delete_files_in_date_range(save_path)

print("File deletion process completed.")
