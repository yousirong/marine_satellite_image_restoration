import os
import shutil
import cv2
import numpy as np
img_path = './test/goci'
save_path = './test/result'

img_list = os.listdir(img_path)

for img in img_list:
    cnt = 0 
    np_a = cv2.imread(img_path+"/"+img, cv2.IMREAD_UNCHANGED)
    for i in range(256):
        for j in range(256):
            if np_a[i,j] < 0:
                np_a[i,j] = 0
    np_a = (np_a-np_a.min())/ (np_a.max()-np_a.min())
    np_a = np_a*255
    cv2.imwrite(save_path + '/'+ img[:-4] +'.png', np_a)



