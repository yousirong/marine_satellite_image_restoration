import os
import shutil
import cv2
import numpy as np
import natsort
from datetime import datetime

goci_path = '/media/pmilab/3dbe7506-c248-4dac-a1f3-866a0bc3ecf8/home/pmimoon/Documents/RFR/data/GOCI_RRS/Rrs_test/2021/gt/3'
goci_save = './goci'
modis_path = '../MODIS/2021/443/0'
modis_save = './modis/2021'


goci_list = os.listdir(goci_path)
goci_list = natsort.natsorted(goci_list)

modis_list = os.listdir(modis_path)
modis_list = natsort.natsorted(modis_list)
idx = 0
'''
for m in modis_list:
    m_date = m[11:19]
    m_date = datetime.strptime(m_date, "%Y%m%d")
    m_hour = m[20:22]
    m_split = m.split('_')
    row = m_split[2]
    m_row = row[1:]
    col = m_split[3]
    m_col = col[1:-5]
    #print(date, hour, row, col)
    for i in range(idx, len(goci_list)):
        g = goci_list[i]
        g_split = g.split('_')
        date = g_split[4]
        g_date = date[:8]
        g_date = datetime.strptime(g_date, "%Y%m%d")
        g_hour = date[9:11]
        row = g_split[5]
        g_row = row[1:]
        col = g_split[6]
        g_col = col[1:-5]

        if m_date < g_date : 
            print(m_date, g_date)
            idx = i
            break

        if m_date == g_date and g_row==m_row and g_col==m_col : 
            print(g, m)
            shutil.copy(os.path.join(goci_path, g), os.path.join(goci_save, g))
            shutil.copy(os.path.join(modis_path, m), os.path.join(modis_save, m))
        else :
            continue
'''

for m in modis_list:
    m_date = m[11:19]
    m_date = datetime.strptime(m_date, "%Y%m%d")
    m_hour = m[20:22]
    m_split = m.split('_')
    row = m_split[2]
    m_row = row[1:]
    col = m_split[3]
    m_col = col[1:-5]
    #print(date, hour, row, col)
    for i in range(len(goci_list)):
        g = goci_list[i]
        g_split = g.split('_')
        date = g_split[4]
        g_date = date[:8]
        g_date = datetime.strptime(g_date, "%Y%m%d")
        g_hour = date[9:11]
        row = g_split[5]
        g_row = row[1:]
        col = g_split[6]
        g_col = col[1:-5]

        if m_date == g_date and g_row==m_row and g_col==m_col and g_hour==m_hour : 
            print(g, m)
            shutil.copy(os.path.join(goci_path, g), os.path.join(goci_save, g))
            shutil.copy(os.path.join(modis_path, m), os.path.join(modis_save, m))
        else :
            continue
