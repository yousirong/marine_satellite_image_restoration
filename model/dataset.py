#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, cv2, re, random
import numpy as np
import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, image_path, mask_path, land_sea_mask_path,
                 mask_mode, target_size, augment=False,
                 training=True, mask_reverse=False):
        super().__init__()
        self.data       = self.load_list(image_path)
        self.mask_data  = self.load_list(mask_path)
        self.target_size= target_size if isinstance(target_size,int) else target_size[0]
        self.mask_reverse = mask_reverse

        # 1) raw .npy → sea_mask: 1=ocean, 0=land
        raw = np.load(land_sea_mask_path)
        print("Raw LS unique:", np.unique(raw))
        self.land_sea_mask = (raw==1).astype(np.uint8)
        print("Converted LS unique (1=ocean,0=land):", np.unique(self.land_sea_mask))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, hole = self.load_item(idx, test_mode=not getattr(self,'training',False))
        if img is None:
            return self.__getitem__((idx+1)%len(self.data))
        fn = os.path.basename(self.data[idx])
        return self.to_tensor(img), self.to_tensor(hole), fn

    def load_item(self, idx, test_mode=False):
        path = self.data[idx]
        # 1) load GT
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None: return None, None
        # to 3ch
        if img.ndim==2:      img = np.stack([img]*3,axis=2)
        elif img.shape[2]==1:img = np.concatenate([img]*3,axis=2)
        elif img.shape[2]==4:img=img[:,:,:3]
        img = img.astype(np.float32)
        img[img==-999]=0
        img = cv2.resize(img,(self.target_size,self.target_size))
        img = img.transpose(2,0,1)  # (3,H,W)

        # 2) load random png-mask 0→1,hole;255→0
        m = cv2.imread(random.choice(self.mask_data),cv2.IMREAD_GRAYSCALE)
        if m is None: m = np.zeros((self.target_size,self.target_size),np.uint8)
        m = cv2.resize(m,(self.target_size,self.target_size))
        mask_png = (m==0).astype(np.uint8)
        mask_png = np.stack([mask_png]*3,axis=0)  # (3,H,W)
        if self.mask_reverse:
            mask_png = 1-mask_png

        # 3) crop land_sea_mask patch
        fn = os.path.basename(path)
        r,c = map(int, re.search(r'r(\d+)_c(\d+)',fn).groups())
        ls = self.land_sea_mask[r:r+self.target_size,
                                c:c+self.target_size]  # (H,W)
        # 4) hole only on ocean
        hole2 = mask_png[0] * ls   # (H,W)
        hole_mask = np.stack([hole2]*3,axis=0).astype(np.uint8)

        # 5) require >1% coverage
        tot = ls.sum()
        if tot>0 and hole2.sum()/tot<0.01:
            return None,None

        return img, hole_mask

    def to_tensor(self, x):
        if isinstance(x,np.ndarray):
            return torch.from_numpy(x.astype(np.float32)/255.0)
        return x

    def load_list(self,path):
        if os.path.isdir(path):
            out=[]
            for r,_,fs in os.walk(path):
                for f in fs:
                    if f.lower().endswith(('.tiff','.png')):
                        out.append(os.path.join(r,f))
            out.sort(); return out
        if os.path.isfile(path):
            return [path]
        return []
