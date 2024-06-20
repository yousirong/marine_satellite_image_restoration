import shutil

f_ = open('./data/places/val.txt', mode='r', encoding='utf-8')
files = f_.readlines()

dir = './data/places/f_val/'
for i, f in enumerate(files):
    print(f[:-1])
    shutil.move('./data/places/'+f[:-1], dir+str(i)+'.jpg')