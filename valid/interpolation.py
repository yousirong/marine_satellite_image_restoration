import cv2
import numpy as np
#from torchvision.utils import save_image
import os 
import glob
def save_img(img, file_path):
        
        #img = img.cpu().numpy()
        img_min = np.min(img)
        img_max = np.max(img)
        
        #min-max normalization
        out_img = (img-img_min)/(img_max-img_min)
        out_img = out_img*255
        
        #img = out_img.cpu().numpy()
        #gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #gray_img = gray_img *255.
        cv2.imwrite(file_path, out_img)


gt_path = "/media/pmilab/3dbe7506-c248-4dac-a1f3-866a0bc3ecf8/home/pmimoon/Documents/RFR/data/GOCI_RRS/Rrs_test/2021/gt/4"
#gt_path = "results/GOCI_RRS_degree/2021/2/10/gt/"
#restored image path
#restored_path = "results/GOCI_RRS_degree/2020/2/50/img/"

#mask image path
#mask_path = "data/GOCI_RRS/Rrs_test/Rrs_mask/1/10/"
mask_path = '/media/pmilab/3dbe7506-c248-4dac-a1f3-866a0bc3ecf8/home/pmimoon/Documents/RFR/data/GOCI_RRS/Rrs_test/2021/Rrs_mask/4/10'

#gt_list = os.listdir(gt_path)
gt_list = list(glob.glob(gt_path + '/*.tiff')) + list(glob.glob(gt_path + '/*.png'))

#restored_list = os.listdir(restored_path)
#mask_list = os.listdir(mask_path)
mask_list = list(glob.glob(mask_path + '/*.tiff')) + list(glob.glob(mask_path + '/*.png'))

gt_list.sort()
#restored_list = sorted(restored_list)

mask_list.sort()

for i in range(len(gt_list)):
    file_part = gt_list[i].split('/')
    file_name = file_part[-1]
    file_name = file_name[:-5]
   
    gt_np =cv2.imread(gt_list[i], cv2.IMREAD_UNCHANGED)
    #gt_np = gt_np/255
    #gt_np = np.loadtxt(gt_path+gt_list[i], delimiter=',',dtype='float32')
    #print(gt_np)
    
    #restored_np = cv2.imread(restored_path+restored_list[i],cv2.IMREAD_UNCHANGED)

    mask = cv2.imread(mask_list[i], cv2.IMREAD_GRAYSCALE)
    #print(mask.shape)
    #mask = np.where(mask>0, 0, 255)
    #mask = 255-mask
    
    #mask_ = mask/255
    masked_np = gt_np*(mask)
    
    dst = cv2.inpaint(masked_np, 255-mask, 3, cv2.INPAINT_TELEA)
    dst = dst/255
    '''
    file_path = '{:s}/gt_img_{:d}.png'.format('./interpolation/rrs', i)
    save_img(gt_np, file_path)
    file_path = '{:s}/masked_img_{:d}.png'.format('./interpolation/rrs', i)
    save_img(masked_np, file_path)
    file_path = '{:s}/dst_{:d}.png'.format('./interpolation/rrs', i)
    save_img(dst, file_path)
    '''
    file_path = '{:s}/{:s}'.format('./interpolation/rrs_test/4/10', file_name)
    np.savetxt(file_path,dst,  delimiter=",")
    #file_path = '{:s}/mask_{:d}'.format('./interpolation/rrs_test_mask/10', i)
    #np.savetxt(file_path,mask/255,  delimiter=",")