import numpy as np
import os
import glob
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import seaborn as sns
import re
from tqdm import trange

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def calculate_pixel_difference(image1, image2):
    # Calculate the difference between the two images
    diff = image2 - image1
    return diff

def normalize_and_apply_colormap(data, vmin=-20, vmax=20, cmap='coolwarm'):
    # Normalize the data to be within the specified range
    norm = Normalize(vmin=vmin, vmax=vmax)
    colormap = cm.ScalarMappable(norm=norm, cmap=cmap)
    colored_data = colormap.to_rgba(data)[:, :, :3]  # Ignore the alpha channel
    return colored_data

def save_difference_image(diff, save_path):
    # Normalize and apply the colormap
    colored_diff = normalize_and_apply_colormap(diff)
    
    # Save the resulting image
    save_path_with_extension = save_path if save_path.lower().endswith('.png') else save_path + '.png'
    plt.imsave(save_path_with_extension, colored_diff)
    
    # Display the image with the updated color bar
    plt.imshow(diff, cmap='coolwarm', norm=Normalize(vmin=-20, vmax=20))
    plt.colorbar(label='Chlorophyll-a concentration difference (mg/m³)')
    plt.title('Chlorophyll-a Concentration Difference')

    # Remove the axis ticks and labels
    plt.xticks([])
    plt.yticks([])
    
    # Save the image with the color bar
    plt.savefig(save_path_with_extension.replace('.png', '_bar.png'), dpi=300, bbox_inches='tight')
    plt.close()

def process_images(recon_path, save_path):
    recon_files_list = sorted(glob.glob(os.path.join(recon_path, '*')), key=natural_sort_key)

    if not recon_files_list:
        print("No image files found in the recon path.")
        return

    diff_image_path = os.path.join(save_path, 'difference_images')
    if not os.path.exists(diff_image_path):
        os.makedirs(diff_image_path)

    for i in trange(1, len(recon_files_list)):
        image1 = np.loadtxt(recon_files_list[i-1], delimiter=',', dtype='float32')
        image2 = np.loadtxt(recon_files_list[i], delimiter=',', dtype='float32')
        
        diff = calculate_pixel_difference(image1, image2)
        
        save_diff_file_name = f'difference_{i-1}_{i}.png'
        save_difference_image(diff, os.path.join(diff_image_path, save_diff_file_name))

# Example usage
data_path = '/path/to/data'
save_path = '/path/to/save'
recon_path = os.path.join(data_path, 'recon')

process_images(recon_path, save_path)
