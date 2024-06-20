import numpy as np
import cv2
from tqdm import tqdm

data_path = '/media/ubuntu/9440c52f-087d-496c-b22b-cd3d455af61b/home/hay/preprocessed/GOCI/Chl-a'
save_path = '/media/ubuntu/9440c52f-087d-496c-b22b-cd3d455af61b/home/hay/preprocessed/GOCI/mask'


# years = sorted(os.listdir(data_path)]
years = [input("year : ")]
losses = ['10','20','30','40','50']


for year in years:
    for loss in losses:
        imgs_path = os.path.join(data_path, year, loss)
        save_loss = os.path.join(save_path, year, loss)
        if not os.path.isdir(save_loss):
            os.makedirs(save_loss)
        imgs = os.listdir(imgs_path)
        for img in tqdm(imgs):
            img_path = os.path.join(imgs_path, img)
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            img = np.nan_to_num(img, nan=0)
            mask = (img==0)
            mask = mask.astype(int)
            mask = 1-mask
            mask = mask*255.0
            cv2.imwrite(os.path.join(save_loss,img+'.png'), mask)#save a .tiff file



