import os
import shutil
def relocate(path_1, path_2, out_path_1, out_path_2):
    filenames_1 = sorted(os.listdir(path_1))
    filenames_2 = sorted(os.listdir(path_2))

    for filename_1 in filenames_1:
        print(filename_1)
        for filename_2 in filenames_2:
            if filename_1 == filename_2:
                shutil.copy(path_1+filename_1, out_path_1+filename_1)
                shutil.copy(path_2+filename_2, out_path_2+filename_2)

def copy_f(path_1, path_2):
    file_1 = sorted(os.listdir(path_1))
    file_1 = file_1[:1890]
    for f in file_1:
        shutil.copy(path_1+f, path_2+f)


for i in range(10, 60, 10):
    #relocate('/home/pmilab/Documents/DayMOM/result_25_140px/mom/'+str(i)+'/01/', '/home/pmilab/Documents/DayMOM/result_25_140px/sst_masked/'+str(i)+'/01/', '/media/pmilab/3dbe7506-c248-4dac-a1f3-866a0bc3ecf8/home/pmimoon/Documents/RFR/data/test/sst/cropDayMOM_140px/01/', '/media/pmilab/3dbe7506-c248-4dac-a1f3-866a0bc3ecf8/home/pmimoon/Documents/RFR/data/test/sst/cropDaySST_mask_140px/01/')
    copy_f("/media/pmilab/3dbe7506-c248-4dac-a1f3-866a0bc3ecf8/home/pmimoon/Documents/RFR/data/GOCI_RRS/Rrs_test/2020/Rrs_mask/3/"+str(i)+"/", "/media/pmilab/3dbe7506-c248-4dac-a1f3-866a0bc3ecf8/home/pmimoon/Documents/RFR/data/GOCI_RRS/train/Rrs_mask/3/")