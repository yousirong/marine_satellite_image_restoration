#model output is named by order
#need to change file name
#order -> original(sst) name
import os
import glob
import shutil

#input test img path
gtpath = "data/train/rrs/380"
#save test result img path
lossimgpath = "data/test/rrs/380"
savepath = "data/test/rrs/380_"
mask_path = "data/test/rrs/mask"
mask_savepath = "data/test/rrs/mask_"
f_list = os.listdir(gtpath)


print(f_list)
for i in range(len(f_list)):
    shutil.copy(lossimgpath+"/"+f_list[i],savepath+"/"+f_list[i])
    shutil.copy(mask_path+"/"+f_list[i],mask_savepath+"/"+f_list[i])
    

