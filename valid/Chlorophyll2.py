from PIL import Image
import cv2
import os
import numpy as np
import glob
from tqdm import trange
import re

# Paths
rrs_gt_path_1 = 'model/results/GOCI_RRS_band2_test/550000/10/gt/'
rrs_gt_path_2 = 'model/results/GOCI_RRS_band3_test/550000/10/gt/'  
rrs_gt_path_3 = 'model/results/GOCI_RRS_band4_test/550000/10/gt/'  

rrs_recon_path_1 = 'model/results/GOCI_RRS_band2_test/550000/10/recon/'
rrs_recon_path_2 = 'model/results/GOCI_RRS_band3_test/550000/10/recon/'
rrs_recon_path_3 = 'model/results/GOCI_RRS_band4_test/550000/10/recon/'

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
mask_files_list = [f for f in glob.glob(os.path.join(rrs_mask, '*'), recursive=True) if os.path.isfile(f)]
gt_rrs1_files_list = [f for f in glob.glob(os.path.join(rrs_gt_path_1, '*'), recursive=True) if os.path.isfile(f)]
gt_rrs2_files_list = [f for f in glob.glob(os.path.join(rrs_gt_path_2, '*'), recursive=True) if os.path.isfile(f)]
gt_rrs3_files_list = [f for f in glob.glob(os.path.join(rrs_gt_path_3, '*'), recursive=True) if os.path.isfile(f)]

recon_rrs1_files_list = [f for f in glob.glob(os.path.join(rrs_recon_path_1, '*'), recursive=True) if os.path.isfile(f)]
recon_rrs2_files_list = [f for f in glob.glob(os.path.join(rrs_recon_path_2, '*'), recursive=True) if os.path.isfile(f)]
recon_rrs3_files_list = [f for f in glob.glob(os.path.join(rrs_recon_path_3, '*'), recursive=True) if os.path.isfile(f)]

# Extract date and coordinates from filenames
def extract_date_coords(filename):
    match = re.search(r'(\d{4}-\d{2}-\d{2})_r(\d+)_c(\d+)', filename)
    if match:
        date = match.group(1)
        r = match.group(2)
        c = match.group(3)
        return date, r, c
    return None, None, None

# Process each file
for i in trange(len(gt_rrs1_files_list)):

    # Extract date and coordinates from GT file
    gt_date, gt_r, gt_c = extract_date_coords(gt_rrs1_files_list[i])
    
    # Filter recon files by matching date and coordinates
    recon_match = [
        recon_file for recon_file in recon_rrs1_files_list
        if extract_date_coords(recon_file) == (gt_date, gt_r, gt_c)
    ]
    
    # If no matching recon file is found, skip this iteration
    if not recon_match:
        continue
    
    img_gt = []
    img_recon = []
    f_name = os.path.basename(gt_rrs1_files_list[i]).replace('.png', '.csv')

    # Load ground truth Rrs bands (using cv2.IMREAD_GRAYSCALE to ensure single channel)
    gt_rrs1 = cv2.imread(gt_rrs1_files_list[i], cv2.IMREAD_GRAYSCALE) / 255
    gt_rrs2 = cv2.imread(gt_rrs2_files_list[i], cv2.IMREAD_GRAYSCALE) / 255
    gt_rrs3 = cv2.imread(gt_rrs3_files_list[i], cv2.IMREAD_GRAYSCALE) / 255
    img_gt.append(gt_rrs1)
    img_gt.append(gt_rrs2)
    img_gt.append(gt_rrs3)

    # Load reconstructed Rrs bands (using cv2.IMREAD_GRAYSCALE to ensure single channel)
    recon_rrs1 = cv2.imread(recon_rrs1_files_list[i], cv2.IMREAD_GRAYSCALE) / 255
    recon_rrs2 = cv2.imread(recon_rrs2_files_list[i], cv2.IMREAD_GRAYSCALE) / 255
    recon_rrs3 = cv2.imread(recon_rrs3_files_list[i], cv2.IMREAD_GRAYSCALE) / 255
    img_recon.append(recon_rrs1)
    img_recon.append(recon_rrs2)
    img_recon.append(recon_rrs3)

    # Load mask (assuming it is grayscale)
    mask = cv2.imread(mask_files_list[i], cv2.IMREAD_GRAYSCALE)

    # Stack Rrs bands for gt and recon
    R_rs_gt = np.stack(img_gt, axis=0)
    R_rs_recon = np.stack(img_recon, axis=0)

    # Now, the shape should be (3, height, width)
    print(f"Shape of R_rs_gt: {R_rs_gt.shape}")
    
    # Check if the shape matches the expected dimensions (3D: channels, height, width)
    if R_rs_gt.ndim != 3:
        print(f"Unexpected shape for R_rs_gt: {R_rs_gt.shape}, skipping...")
        continue

    _, height, width = R_rs_gt.shape

    a = [0.2515, -2.3798, 1.5823, -0.6372, -0.5692]
    Chl_oc3_gt = np.empty((height, width))
    Chl_oc3_recon = np.empty((height, width))

    for h in range(height):
        for w in range(width):
            # Ground truth chlorophyll synthesis
            if R_rs_gt[2, h, w] <= 0 or R_rs_gt[0, h, w] <= 0 or R_rs_gt[1, h, w] <= 0:
                Chl_oc3_gt[h, w] = 0
            else:
                term = np.sum(a[i] * (np.log10(np.max(R_rs_gt[:2, h, w]) / R_rs_gt[2, h, w]))**i for i in range(1, 5))
                Chl_oc3_gt[h, w] = 10 ** (a[0] + term)

            # Reconstructed chlorophyll synthesis
            if R_rs_recon[2, h, w] <= 0 or R_rs_recon[0, h, w] <= 0 or R_rs_recon[1, h, w] <= 0:
                Chl_oc3_recon[h, w] = 0
            else:
                term = np.sum(a[i] * (np.log10(np.max(R_rs_recon[:2, h, w]) / R_rs_recon[2, h, w]))**i for i in range(1, 5))
                Chl_oc3_recon[h, w] = 10 ** (a[0] + term)

    # Save the computed chlorophyll concentration and mask as CSV files
    np.savetxt(os.path.join(gt_save_path, 'gt_' + f_name), Chl_oc3_gt, delimiter=',')
    np.savetxt(os.path.join(recon_save_path, 'recon_' + f_name), Chl_oc3_recon, delimiter=',')
    np.savetxt(os.path.join(mask_save_path, 'mask_' + f_name), mask, delimiter=',')
