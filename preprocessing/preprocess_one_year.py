# UST21 Chl-a 데이터의 특정 년도에 대한 정보 추출 및 전처리
import h5py as h5
import netCDF4 as nc
import numpy as np
import os
import cv2
import math
from tqdm import tqdm


def check_pct(arr):
    nans = np.isnan(arr)
    zeros = (arr == 0)
    neg_outlier = (arr < 0)
    pos_outlier = (arr>20)
    assert np.sum(pos_outlier) <1
    count = np.sum(nans) + np.sum(zeros) + np.sum(neg_outlier) +np.sum(pos_outlier)
    #loss rate
    pct = count/(256*256) *10
    temp_pct = pct*10
    pct = math.floor(pct)*10
    return temp_pct, pct


########## path ##########
data_base = '/media/ubuntu/My Book/UST21/Daily'
save_base = '/media/ubuntu/My Book/data/Chl-a'
ocean_base = '/media/ubuntu/My Book/data/Chl-a/ocean'
ocean_idx = np.load("/home/ubuntu/문서/AY_ust/preprocessing/ocean_idx_arr.npy")
######### point #########
x_nak, y_nak = (3377, 3664)
x_sae, y_sae = (3751, 4629)
##########################


year = str(input())
months = [f"{i:02}" for i in range(1, 13)]
pcts= [str(i) for i in range(0, 100, 10)]
pcts.append('perfect')


for phase in ['train', 'test', 'ocean']:
    for pct in pcts:
        temp = os.path.join(save_base, phase, pct)
        if not os.path.isdir(temp):
            os.makedirs(temp)


phase = 'train' if int(year)<2021 else 'test'
data_year = os.path.join(data_base, year)
for month in tqdm(months):
    data_month = os.path.join(data_year, str(month))
    if os.path.isdir(data_month):
        imgs = os.listdir(data_month)
        for img in imgs:
            if not 'nc' in img:
                continue
            path = os.path.join(data_month, img)
            f = nc.Dataset(path,'r')
            a = f['merged_daily_Chl'][:].data
            np_a = np.array(a)
            np_a = np.where(np_a==-999.0, 0, np_a)
            np_a[np_a>20]=0
            idx = 0
            row, col = np_a.shape

            #중요 해역 추출
            n_patch = np_a[x_nak:x_nak+256, y_nak:y_nak+256]
            s_patch = np_a[x_sae:x_sae+256, y_sae:y_sae+256]

            s_temp , s_pct = check_pct(s_patch)
            n_temp , n_pct = check_pct(n_patch)

            if s_pct == 100 :
                pass
            else:
                if s_temp<1:
                    save_path = os.path.join(ocean_base, 'perfect', img[:-3]+'_sae.tiff')
                    cv2.imwrite(save_path, s_patch)
                save_path = os.path.join(ocean_base, str(s_pct), img[:-3]+'_sae.tiff')
                cv2.imwrite(save_path, s_patch)

            if n_pct == 100:
                continue
            else:
                if n_temp<1:
                    save_path = os.path.join(ocean_base, 'perfect', img[:-3]+'_nak.tiff')
                    cv2.imwrite(save_path, n_patch)
                save_path = os.path.join(ocean_base, str(n_pct), img[:-3]+'_nak.tiff')
                cv2.imwrite(save_path, n_patch)

            #전체 훈련 샘플 추출
            for k in range(0, row, 256):
                for r in range(0, col, 256):
                    if idx in ocean_idx:
                        new_arr = np_a[k:k+256, r:r+256]

                        if new_arr.shape!=(256,256):
                            continue
                        row_col = '_r' + str(k) + '_c' + str(r)

                        #outliers
                        temp_pct, pct = check_pct(new_arr)

                        #save file
                        if pct == 100 :
                            continue
                        if temp_pct<1:
                            save_path = os.path.join(save_base, phase, 'perfect', img[:-3] + row_col+'.tiff')
                            cv2.imwrite(save_path, new_arr )#save a .tiff file
                        # np.savetxt(save_path+'/'+ y+'/' +str(pct)+'/'+ i[:-7] + row_col, new_arr, delimiter=",") # save a .txt file
                        save_path = os.path.join(save_base, phase, str(pct), img[:-3] + row_col+'.tiff')
                        cv2.imwrite(save_path, new_arr )#save a .tiff file
                    idx = idx+1
            f.close()

