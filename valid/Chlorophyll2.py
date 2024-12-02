from PIL import Image
import cv2
import os
import numpy as np
import glob
from tqdm import tqdm
import re

# Paths
rrs_gt_path_1 = '/home/juneyonglee/MyData/backup_20240914/AY_UST/results/GOCI_RRS_band2_test/550000/10/gt/'
rrs_gt_path_2 = '/home/juneyonglee/MyData/backup_20240914/AY_UST/results/GOCI_RRS_band3_newtrain_test/550000/10/gt/'
rrs_gt_path_3 = '/home/juneyonglee/MyData/backup_20240914/AY_UST/results/GOCI_RRS_band4_test/550000/10/gt/'

rrs_recon_path_1 = '/home/juneyonglee/MyData/backup_20240914/AY_UST/results/GOCI_RRS_band2_test/550000/10/recon/'
rrs_recon_path_2 = '/home/juneyonglee/MyData/backup_20240914/AY_UST/results/GOCI_RRS_band3_newtrain_test/550000/10/recon/'
rrs_recon_path_3 = '/home/juneyonglee/MyData/backup_20240914/AY_UST/results/GOCI_RRS_band4_test/550000/10/recon/'

rrs_mask = '/media/juneyonglee/My Book/data/mask/Test/10'
save_path = 'model/results/GOCI_chl/550000/10'

# Create degree directory and subdirectories for gt, mask, and recon
degree_path = os.path.join(save_path, 'degree')
gt_save_path = os.path.join(degree_path, 'gt')
mask_save_path = os.path.join(degree_path, 'mask')
recon_save_path = os.path.join(degree_path, 'recon')

# Create the directories if they don't exist
for path in [degree_path, gt_save_path, mask_save_path, recon_save_path]:
    if not os.path.isdir(path):
        os.makedirs(path)

# Load files and filter out directories
gt_rrs1_files_list = sorted(glob.glob(os.path.join(rrs_gt_path_1, '*')), key=os.path.basename)
gt_rrs2_files_list = sorted(glob.glob(os.path.join(rrs_gt_path_2, '*')), key=os.path.basename)
gt_rrs3_files_list = sorted(glob.glob(os.path.join(rrs_gt_path_3, '*')), key=os.path.basename)

recon_rrs1_files_list = sorted(glob.glob(os.path.join(rrs_recon_path_1, '*')), key=os.path.basename)
recon_rrs2_files_list = sorted(glob.glob(os.path.join(rrs_recon_path_2, '*')), key=os.path.basename)
recon_rrs3_files_list = sorted(glob.glob(os.path.join(rrs_recon_path_3, '*')), key=os.path.basename)

mask_files_list = sorted(glob.glob(os.path.join(rrs_mask, '*')), key=os.path.basename)

# Extract date and coordinates from filenames
def extract_date_coords(filename):
    match = re.search(r'(\d{4}-\d{2}-\d{2})_r(\d+)_c(\d+)', filename)
    if match:
        date = match.group(1).replace('-', '')  # Convert to YYYYMMDD format
        r = match.group(2)
        c = match.group(3)
        return date, r, c
    return None, None, None

# Check if the file path contains "nak" or "sae" and return as a region
def extract_region(file_path):
    if "nak" in file_path:
        return "nak"
    elif "sae" in file_path:
        return "sae"
    return ""

# Process matching files
for i in tqdm(range(len(gt_rrs1_files_list))):
    gt_date1, gt_r1, gt_c1 = extract_date_coords(gt_rrs1_files_list[i])
    region = extract_region(gt_rrs1_files_list[i])  # Determine region from path

    # Find matching files in gt_rrs2_files_list and gt_rrs3_files_list based on date, row, and column
    gt_rrs2_match = [f for f in gt_rrs2_files_list if extract_date_coords(f) == (gt_date1, gt_r1, gt_c1)]
    gt_rrs3_match = [f for f in gt_rrs3_files_list if extract_date_coords(f) == (gt_date1, gt_r1, gt_c1)]

    if not gt_rrs2_match or not gt_rrs3_match:
        continue

    gt_rrs2_file = gt_rrs2_match[0]
    gt_rrs3_file = gt_rrs3_match[0]

    # Find matching files in recon_rrs1, recon_rrs2, recon_rrs3
    recon_rrs1_match = [f for f in recon_rrs1_files_list if extract_date_coords(f) == (gt_date1, gt_r1, gt_c1)]
    recon_rrs2_match = [f for f in recon_rrs2_files_list if extract_date_coords(f) == (gt_date1, gt_r1, gt_c1)]
    recon_rrs3_match = [f for f in recon_rrs3_files_list if extract_date_coords(f) == (gt_date1, gt_r1, gt_c1)]

    if not recon_rrs1_match or not recon_rrs2_match or not recon_rrs3_match:
        continue

    recon_rrs1_file = recon_rrs1_match[0]
    recon_rrs2_file = recon_rrs2_match[0]
    recon_rrs3_file = recon_rrs3_match[0]

    img_gt, img_recon = [], []

    # Load ground truth Rrs bands
    gt_rrs1 = cv2.imread(gt_rrs1_files_list[i], cv2.IMREAD_GRAYSCALE) / 255
    gt_rrs2 = cv2.imread(gt_rrs2_file, cv2.IMREAD_GRAYSCALE) / 255
    gt_rrs3 = cv2.imread(gt_rrs3_file, cv2.IMREAD_GRAYSCALE) / 255
    img_gt.extend([gt_rrs1, gt_rrs2, gt_rrs3])

    # Load reconstructed Rrs bands
    recon_rrs1 = cv2.imread(recon_rrs1_file, cv2.IMREAD_GRAYSCALE) / 255
    recon_rrs2 = cv2.imread(recon_rrs2_file, cv2.IMREAD_GRAYSCALE) / 255
    recon_rrs3 = cv2.imread(recon_rrs3_file, cv2.IMREAD_GRAYSCALE) / 255
    img_recon.extend([recon_rrs1, recon_rrs2, recon_rrs3])

    mask = cv2.imread(mask_files_list[i], cv2.IMREAD_GRAYSCALE)

    # Stack Rrs bands for gt and recon
    R_rs_gt = np.stack(img_gt, axis=0)
    R_rs_recon = np.stack(img_recon, axis=0)

    if R_rs_gt.ndim != 3:
        print(f"Unexpected shape for R_rs_gt: {R_rs_gt.shape}, skipping...")
        continue

    _, height, width = R_rs_gt.shape

    a = [0.2515, -2.3798, 1.5823, -0.6372, -0.5692]
    Chl_oc3_gt = np.empty((height, width))
    Chl_oc3_recon = np.empty((height, width))

    for h in range(height):
        for w in range(width):
            if R_rs_gt[2, h, w] <= 0 or R_rs_gt[0, h, w] <= 0 or R_rs_gt[1, h, w] <= 0:
                Chl_oc3_gt[h, w] = 0
            else:
                term = sum(a[i] * (np.log10(np.max(R_rs_gt[:2, h, w]) / R_rs_gt[2, h, w]))**i for i in range(1, 5))
                Chl_oc3_gt[h, w] = 10 ** (a[0] + term)

            if R_rs_recon[2, h, w] <= 0 or R_rs_recon[0, h, w] <= 0 or R_rs_recon[1, h, w] <= 0:
                Chl_oc3_recon[h, w] = 0
            else:
                term = sum(a[i] * (np.log10(np.max(R_rs_recon[:2, h, w]) / R_rs_recon[2, h, w]))**i for i in range(1, 5))
                Chl_oc3_recon[h, w] = 10 ** (a[0] + term)

    # Create a filename with region and formatted date
    file_suffix = f"{gt_date1}_{region}_r{gt_r1}_c{gt_c1}"

    # Save chlorophyll concentration and mask as CSV files
    np.savetxt(os.path.join(gt_save_path, f'gt_{file_suffix}.csv'), Chl_oc3_gt, delimiter=',')
    np.savetxt(os.path.join(recon_save_path, f'recon_{file_suffix}.csv'), Chl_oc3_recon, delimiter=',')
    np.savetxt(os.path.join(mask_save_path, f'mask_{file_suffix}.csv'), mask, delimiter=',')
