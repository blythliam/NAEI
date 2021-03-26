# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 16:09:17 2021

@author: lpb20
"""
#Importing required modules

import glob
import numpy as np
import pandas as pd
import time
import os
from scipy import ndimage

#%%
" Setting variables depending on desired outputs "

file_write = True
spat_ocean = False


#%%
" Loading file paths to be used later in routine "

root_path = 'C:/Users/lpb20/OneDrive - Imperial College London/Documents/Odyssey/NAEI'
area_grid_cell_path =  'C:/Users/lpb20/OneDrive - Imperial College London/Documents/Odyssey/Footprints'

map_path =root_path + '/emission_maps'

footprint_path = 'C:/Users/lpb20/OneDrive - Imperial College London/Documents/Odyssey/Footprints/2km_footprints/IMP_26magl/Fields_files'

if spat_ocean:
    os.chdir('//home/lpb20/NAEI')
    
    area_grid_cell_path = './area_file'
    footprint_path =  './Fields_files'
    map_path = './emission_maps'



#%%
" Importing 2km area file"

area_grid_cell = np.loadtxt(area_grid_cell_path + '/area_2km.txt')

print('----------------------------------')
print('\tArea Files have been produced')
print('----------------------------------')


#%%
" Imprting NAEI mapped data (Lat-Lon) and converting to kg "

files = glob.glob(map_path + '/*.txt')

mapped_data = np.zeros([627, 767, len(files)])

for i, file in enumerate(files):
    mapped_data[:,:,i] = np.loadtxt(file, delimiter=',')
    
print('------------------------------------')
print('\tAll mapped data has been loaded')
print('------------------------------------')

#%%


p = 1204.1 # Density of dry air
mol_air = 28.97 # Molar mass of dry air
mol_CO = 28 # Molar mass of CO


#%%


files = glob.glob(footprint_path + '/*.txt')

days_num = len(files)

XCO = np.zeros([24*days_num,10])
CO_g_m3 = np.zeros([24 * days_num,10])
C_of_M = np.zeros([24 * days_num,2])
tot_14C = np.zeros([627,767])
tot_14C_count = np.zeros([627,767])

start = time.time()

for z, file in enumerate(files):
    if z%5 == 0 :
        print(days_num-z)
    f = open(file)
    footprint = np.zeros([24,627,767])

        
    for i, line in enumerate(f):
        
        if i>25:
            columns = line.split(',')
            for ii in range(24):
                x = int(float(columns[0]) -1.5)
                y = int(float(columns[1])-1.5)
                footprint[ii,y,x] = float(columns[ii+ 4])

    for j in range(24):
                
        for t in range(10):
            C_of_M[(z*24 + j),0] = ndimage.measurements.center_of_mass(footprint[j,:,:])[0]-110
            C_of_M[(z*24 + j),1] = ndimage.measurements.center_of_mass(footprint[j,:,:])[1]-515

            M = mapped_data[:,:,t] * footprint[j,:,:] * area_grid_cell / 3600
            
            CO_g_m3[(z*24 + j), t] = np.sum(M)
            XCO[z*24 + j, t] = (np.sum((M / p) * (mol_air/mol_CO) * 1e9))
            
    
end = time.time()
print('----------------------------------')
print('\t' + str(int(len(files)/30)) + ' months of analysis took ' + str(int(((end - start)/60))) + ' mins')
print('----------------------------------')

#%%

if file_write:
    
    np.savetxt(root_path + '/COmixingratio.csv', XCO, delimiter=',')
    np.savetxt(root_path + '/C_of_M.csv',C_of_M, delimiter=',')

