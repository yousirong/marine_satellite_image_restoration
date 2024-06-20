#model output is named by order
#need to change file name
#order -> original(sst) name
import os
import glob

#input test img path
sstPath = "data/GOCI_RRS/Rrs_test/2021/gt/4/"

#save test result img path
resultPath = "/home/pmilab/Documents/validation_data/recon/GOCI_RRS_degree/2021/4/10/img/"


f_list = list(glob.glob(sstPath+'*.tiff'))
f_list.sort()

result_list = list(glob.glob(resultPath+'*'))
#result_list = [i for i in result_list if "masked" not in i]
result_list.sort()

print("f_list len", len(f_list))
print("result_list len", len(result_list))

for i in range(len(result_list)):
    #print(f_list[i])#crop_20180331.daily.cf.sst.tif
    #print(result_list[i])#img_99.png
    if not os.path.isdir(f_list[i]):
        filename = os.path.splitext(f_list[i])
        #print(filename)
        os.rename(resultPath+"img_"+str(i+1), resultPath + filename[0].split('/')[-1])
        #print(resultPath + filename[0].split('/')[-1]+'.tif') 
        #print(resultPath+"img_"+str(i+1)+".png")
        #print(resultPath + filename[0].split('/')[1]+'.tif')

    
