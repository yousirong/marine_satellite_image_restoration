from scipy import io
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np


mat_file = io.loadmat('/home/ubuntu/문서/AY_ust/preprocessing/Land_mask/Land_mask.mat')
save_file = '/home/ubuntu/문서/AY_ust/preprocessing/ocean_idx_arr.npy'
mask = mat_file['Land']

image_size = mask.shape
patch_size = 256

image_height, image_width = mask.shape[0], mask.shape[1]

num_patches_x = image_width // 256
num_patches_y = image_height // 256

# 패치들을 저장할 NumPy 배열을 초기화합니다. 패치의 개수와 크기에 맞춰 배열의 크기를 설정합니다.
# 이미지가 컬러라면 3차원, 흑백이라면 2차원일 것입니다. 여기서는 컬러 이미지를 가정합니다.
patches = np.empty((num_patches_y * num_patches_x, 256, 256), dtype=mask.dtype)

# 각 패치를 추출하여 NumPy 배열에 추가합니다.
for i in range(num_patches_y):
    for j in range(num_patches_x):
        # 패치의 시작점을 계산합니다.
        start_x = j * 256
        start_y = i * 256
        # NumPy 배열의 인덱스를 계산합니다.
        index = i * num_patches_x + j
        # 이미지에서 해당 패치를 추출하고 배열에 저장합니다.
        patches[index] = mask[start_y:start_y + 256, start_x:start_x + 256]

patch_sums = patches.reshape(num_patches_y * num_patches_x, -1).sum(axis=1)

ocean_index = np.where(patch_sums==0)[0]

np.save(save_file, ocean_index)