# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 21:33:08 2021

@author: lpb20
"""

import glob
import numpy as np
import pandas as pd
import time
import os
from scipy import ndimage
import matplotlib.pyplot as plt

#%%
" Setting variables depending on desired outputs "

plot = True
D14_plot = True
file_write = True
spat_ocean = False


#%%
" Loading file paths to be used later in routine "

root_path = 'C:/Users/lpb20/OneDrive - Imperial College London/Documents/Odyssey/NAEI'
area_grid_cell_path =  'C:/Users/lpb20/OneDrive - Imperial College London/Documents/Odyssey/Footprints'

map_path =root_path + '/emission_maps'

footprint_path = 'C:/Users/lpb20/OneDrive - Imperial College London/Documents/Odyssey/Footprints/2km_footprints/IMP_26magl/Fields_files'


files = glob.glob(footprint_path + '/*.txt')

days_num = len(files)

XCO = np.zeros([24*days_num,10])
CO_g_m3 = np.zeros([24 * days_num,10])
C_of_m = np.zeros([24 * days_num,2])
tot_14C = np.zeros([627,767])
tot_14C_count = np.zeros([627,767])

start = time.time()

for z, file in enumerate(files):
    if z == 0 :
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
footprint_test = footprint[0,:,:]

C_of_M = ndimage.measurements.center_of_mass(footprint_test)
plt.imshow(footprint_test)
plt.plot(C_of_M[1],C_of_M[0],'rx')
    
    
#%%



XCO = np.sum(np.loadtxt(root_path + '/COmixingratio.csv', delimiter = ','),axis = 1)


C_of_M = np.loadtxt(root_path + '/C_of_M.csv', delimiter = ',')

data_to_pd = {'E--W':C_of_M[:,1], 'S--N':C_of_M[:,0],'CO enhancments':XCO}



import seaborn as sns
data = pd.DataFrame.from_dict(data_to_pd)

sns.scatterplot(data = data, x = 'E--W', y = 'S--N',hue = 'CO enhancments')

plt.plot([-300,100],[0,0],'r')
plt.plot([0,0],[-50,250],'r')

data['Dist'] = np.sqrt(data['E--W'] **2 + data['S--N']**2)
data_NW = data.loc[(data['E--W'])<0]
data_NW = data_NW.loc[(data['S--N']>0)]
data_SW = data.loc[(data['E--W'])<0]
data_SW = data_SW.loc[(data['S--N']<0)]
data_NE = data.loc[(data['E--W'])>0]
data_NE = data_NE.loc[(data['S--N']>0)]
data_SE = data.loc[(data['E--W'])>0]
data_SE = data_SE.loc[(data['S--N']<0)]

plt.figure()
sns.scatterplot(data = data_SW, x = 'E--W', y = 'S--N',hue = 'CO enhancments')

#%%
fig = plt.subplot()

for i in range(4):
    if i == 0: data = data_NW; title = 'N--W'
    elif i ==1: data = data_NE; title = 'N--E'
    elif i == 2: data = data_SW; title = 'S--W'
    elif i ==3: data = data_SE; title = 'S--E'
        
    plt.subplot(2,2,i+1)
    plt.plot(data['Dist'], data['CO enhancments'],'x')
    plt.title(title)
    if i == 0 or i ==2:
        plt.ylabel('CO enhancment (ppb)')
    if i == 2 or i == 3:
        plt.xlabel('Distance from Imperial')
    #sns.scatterplot(data, x = 'E--W', y = 'S--N',hue = 'CO enhancments')

#%%
fig = plt.subplot()

for i in range(4):
    if i == 0: data = data_NW; title = 'N--W'
    elif i ==1: data = data_NE; title = 'N--E'
    elif i == 2: data = data_SW; title = 'S--W'
    elif i ==3: data = data_SE; title = 'S--E'
        
    plt.subplot(2,2,i+1)
    sns.boxplot(data['CO enhancments'],orient = 'v')
    plt.text(np.median(data['CO enhancments']) + 20,0.4,
             'CO enh = '+str(round(np.median(data['CO enhancments']),1))
             +' $\pm$ '+str(round(np.std(data['CO enhancments']),1)))
#    plt.title(title)
    if i == 0 or i ==2:
        plt.ylabel('CO enhancment (ppb)')
    if i == 2 or i == 3:
        plt.xlabel('Distance from Imperial')















    
    
    
    
    
    