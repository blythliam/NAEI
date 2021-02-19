# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 14:31:48 2021

@author: lpb20
"""

# COnverting ASCii file to txt/csv file for NAEI emissions

import numpy as np
import glob
import re

input_path = 'C:/Users/lpb20/OneDrive - Imperial College London/Documents/Odyssey/NAEI/Lat_Lon_4'

output_path = 'C:/Users/lpb20/OneDrive - Imperial College London/Documents/Odyssey/NAEI/emission_maps'

files = glob.glob(input_path + '/*.asc')

em_map = np.zeros([627,767, len(files)])
for i,file in enumerate(files):

    
    f = open(file,'r')
    for j,line in enumerate(f):
        if j>5:
            columns = line.split()
            columns = [float(val) for val in columns]
            em_map[j-6,:,i] = columns
    em_map[em_map<0] = 0
up_data_maps = np.zeros([627,767,len(files)])    

for i in range(len(files)):
    down_data = em_map[:,:,i]
    up_data = np.flipud(down_data)
    up_data_maps[:,:,i] = up_data
    
for i in range(len(files)):
    file = files[i]
    text = file
    m = re.search("(\w+?).asc", text)
    files_ext = m.group(1)
    print(files_ext)
    np.savetxt(output_path + '/' +  files_ext +'.txt',up_data_maps[:,:,i], delimiter=',')






