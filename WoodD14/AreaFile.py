# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 16:09:17 2021

@author: lpb20
"""
#Importing required modules
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#%%

area_grid_cell_path =  'C:/Users/lpb20/OneDrive - Imperial College London/Documents/Odyssey/Footprints'

area_grid_cell = np.loadtxt(area_grid_cell_path + '/area_2km.txt')

print('----------------------------------')
print('\tArea Files have been produced')
print('----------------------------------')


#%%


# Outputting array as a csv file
files = glob.glob('C:/Users/lpb20/OneDrive - Imperial College London/Documents/Odyssey/NAEI/Temp/TOTAL.asc')
mapped_data_tot=np.zeros([627,767])
for i,file in enumerate(files):
    

    f = open(file,'r')
    for j,line in enumerate(f):
        if j>5:
            columns = line.split()
            columns = [float(i) for i in columns]
            mapped_data_tot[j-6,:] = columns
            
mapped_data_tot[mapped_data_tot<0] = 0

import re

files = glob.glob('C:/Users/lpb20/OneDrive - Imperial College London/Documents/Odyssey/NAEI/Lat_Lon_4/*.asc')
mapped_data=np.zeros([627,767,10])
for i,file in enumerate(files):
    
    text = file
    m = re.search("(\w+?)co18.asc", text)
    print(m.group(1))
    f = open(file,'r')
    for j,line in enumerate(f):
        if j>5:
            columns = line.split()
            columns = [float(val) for val in columns]
            mapped_data[j-6,:,i] = columns
    mapped_data[mapped_data<0] = 0

print('----------------------------------')
print('\tAll mapped data has been loaded')
print('----------------------------------')

#%%

file ='C:/Users/lpb20/OneDrive - Imperial College London/Documents/Odyssey/NAEI/Temp/footprint.asc'


footprint_test = np.zeros([627,767])
f = open(file,'r')
for i, line in enumerate(f):
    if i>5:
        columns = line.split()
        columns = [float(i) for i in columns]
        footprint_test[i-6] = columns
        footprint_test[footprint_test<0] = 0








#%%

XCO = np.zeros([744,10])
CO_g_m3 = np.zeros([744,10])

import glob
from scipy import ndimage
big_val = {}
files = glob.glob('C:/Users/lpb20/OneDrive - Imperial College London/Documents/Odyssey/Footprints/2km_footprints/IMP_26magl/Fields_files/*.txt')
#footprint_file = 'C:/Users/lpb20/OneDrive - Imperial College London/Documents/Odyssey/Footprints/2km_footprints/IMP_26magl/Fields_files/Fields_grid_A_20171231.txt'
days_num = len(files)

XCO = np.zeros([24*days_num,10])
CO_g_m3 = np.zeros([24 * days_num,10])
centre_of_mass = np.zeros([24 * days_num,3])

import time
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
                x = int(np.floor(float(columns[0])) - 1)
                y = int(np.ceil(627 * (1 -(float(columns[1])/627))))
                footprint[ii,y,x] = float(columns[ii+ 4])

    for j in range(24):
        centre_of_mass[(z*24 + j),0] = ndimage.measurements.center_of_mass(footprint[j,:,:])[0]
        centre_of_mass[(z*24 + j),1] = ndimage.measurements.center_of_mass(footprint[j,:,:])[1]
        
        for t in range(10):
            M = mapped_data[:,:,t] * footprint[j,:,:] * area_grid_cell / 3600
    
            CO_g_m3[(z*24 + j), t] = np.sum(M)
            XCO[z*24 + j, t] = (np.sum((M / 1200) * (28.97/30) * 1e9))
end = time.time()
print('----------------------------------')
print('\t' + str(int(len(files)/30)) + ' months of analysis took ' + str(int(((end - start)/60))) + ' mins')
print('----------------------------------')

#%%
#centre_of_mass[:,2] = np.sum(XCO, axis = 1)
centre_of_mass = pd.DataFrame(centre_of_mass, columns = ['x','y','values'])
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





#%%

#plus14CO = (np.array(CO_g_m3) * 1.176e-12 * 0.25 * 1e-4)*6.02e23/28

D14 = np.array([-329,-784,-787,-1000,22,-1000,-1000,-1000,-33.7,-463])

factor = (1000 + D14)*1e-3 * 1.176e-12

D14CO = CO_g_m3 * factor * 1e-4 * 6.02e23/30
D14CO_sum = np.sum(D14CO, axis = 1)
plt.figure()
sns.histplot(D14CO_sum)
plt.xlabel('$^{14}$CO (molecules cm$^{-3}$)')











  
    



