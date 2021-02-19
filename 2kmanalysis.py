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

#%%
" Setting variables depending on desired outputs "

plot = True
D14_plot = False
file_write = False
spat_ocean = False


#%%
" Loading file paths to be used later in routine "
# Path to 
area_grid_cell_path =  'C:/Users/lpb20/OneDrive - Imperial College London/Documents/Odyssey/Footprints'

map_path ='C:/Users/lpb20/OneDrive - Imperial College London/Documents/Odyssey/NAEI/emission_maps'

footprint_path = 'C:/Users/lpb20/OneDrive - Imperial College London/Documents/Odyssey/Footprints/2km_footprints/IMP_26magl/Fields_files'

if spat_ocean == True:
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



start = time.time()

for z, file in enumerate(files):
    if z%5 == 0 :
        print(len(files)-z)
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
            M = mapped_data[:,:,t] * footprint[j,:,:] * area_grid_cell / 3600
    
            CO_g_m3[(z*24 + j), t] = np.sum(M)
            XCO[z*24 + j, t] = (np.sum((M / p) * (mol_air/mol_CO) * 1e9))

end = time.time()
print('----------------------------------')
print('\t' + str(int(len(files)/30)) + ' months of analysis took ' + str(int(((end - start)/60))) + ' mins')
print('----------------------------------')

#%%
#centre_of_mass[:,2] = np.sum(XCO, axis = 1)

if file_write == True:
    
    np.savetxt('output.csv', XCO, delimiter=',')


if plot == True:
    import matplotlib.pyplot as plt
    import seaborn as sns

   
    sns.histplot(np.sum(XCO, axis = 1), binwidth=5)
    plt.xlabel('CO (ppb)')
    
    plt.figure()
    columns = ['Dom','enprod','indcom','indprod','nature','offshore','othertrans',
               'road_trans','solvent','waste']
    
    box_data = pd.DataFrame(XCO *100 / np.sum(XCO,axis = 1).reshape([24 * days_num,1]),columns = columns)
    box_data.boxplot()
    
    columns_em_perc = np.array([29,5.5,13,7,2.5,0.05,27.7,13.5,0.12,2])
    
    for i,val in enumerate(columns_em_perc):
        plt.plot([i+0.5,i+1.5],[val,val],'r--')

    
    plt.ylabel('% of CO from sector')
    plt.xlabel('SNAP sector')
    if D14_plot == True:
        D14 = np.array([-329,-784,-787,-1000,22,-1000,-1000,-1000,-33.7,-463])
        
        factor = (1000 + D14)*1e-3 * 1.176e-12
        
        D14CO = CO_g_m3 * factor * 1e-4 * 6.02e23/30
        D14CO_sum = np.sum(D14CO, axis = 1)
        plt.figure()
        sns.histplot(D14CO_sum)
        plt.xlabel('$^{14}$CO (molecules cm$^{-3}$)')











  
    



