import os
base= '/home/pmilab/Documents/preprocessed_data/GOCI/Rrs'
band = '2'
train = True
years = os.listdir(base)
cnt = 0
for year in years:
    band_path = os.path.join(base, year,band,'0')
    cnt += len(os.listdir(band_path))


print(cnt)

