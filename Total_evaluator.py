# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 09:08:50 2020

@author: lpb20
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import netCDF4 as nc


#emis_fact = pd.read_csv('C:/Users/lpb20/OneDrive - Imperial College London/Documents/Odyssey/NAEI/emissionfactors.txt',sep = '\t',index_col=0,header=None)




files = glob.glob('C:/Users/lpb20/Documents/NAEI/4/*.asc')
files[0], files[1] = files[1], files[0]
files[4], files[5] = files[5], files[4]
files[5], files[8] = files[8], files[5]
files[6], files[7] = files[7], files[6]
files[7], files[8] = files[8], files[7]
files[9], files[11] = files[11], files[9]
files[7], files[8] = files[8], files[7]
files[8], files[9] = files[9], files[8]
sector_prod=[]
sep_maps = np.zeros([1378,812,12])
sep_maps_14 = np.zeros([1378,812,12])
sep_maps_no14 = np.zeros([1378,812,12])
for j,file in enumerate(files):

    mapped_data=np.zeros([1378,812])
#    print(j,emit_fact,file)
    f = open(file,'r')
    for i,line in enumerate(f):
        if i>5:
            columns = line.split()
            columns = [float(i) for i in columns]
            mapped_data[i-6,:] = columns
    mapped_data[mapped_data<0] = 0
    sep_maps[:,:,j] = mapped_data


sector_prod = np.sum(np.sum(sep_maps, axis = 0),axis = 0)/1e3
#%%
import re

secs = []

for file in files:
    text = file
    m = re.search("(\w+?)co18.asc", text)
    secs.append(m.group(1))
    
    
sector_prod_s = pd.Series(sector_prod) / 1000
sector_prod_s.index = secs

sector_prod_s_perc = sector_prod_s*100 / sector_prod_s['total']



#%%
import matplotlib
from matplotlib.pyplot import contour, pcolor
road_data = sep_maps[:,:,6] / sep_maps[:,:,11]

test = np.logical_and(sep_maps[:,:,1],sep_maps[:,:,11])

road_prom = np.zeros([1378,812])
for i in range(1378):
    for j in range(812):
        if test[i,j] == True:
            road_prom[i,j] = sep_maps[:,:,1][i,j] / sep_maps[:,:,11][i,j]
        else:
            road_prom[i,j] = 0


np.nan_to_num(road_data)
x = np.linspace(0,812,812)
y = np.linspace(0,1378,1378)



#mapped_data[mapped_data>nintyperc] = nintyperc
import matplotlib.colors as mcolors
colors = ["White", "Green", "Yellow", "Orange", "Red"]
cmap = mcolors.LinearSegmentedColormap.from_list("", colors)
#cmap = plt.cm.jet
#white = np.ones((1,4))
#upper = cmap(np.linspace(1-(255/256),1,255))
#colors = np.vstack((white,upper))
#tmap = matplotlib.colors.LinearSegmentedColormap.from_list('jet', colors)

#%%

final_14_1 = final_14 * 3.175e-15

nintyperc = np.percentile(road_prom,99)#final_14_1,95)

pcolor(x,y,road_prom,cmap = cmap, vmin = 0, vmax = nintyperc)
cbar = plt.colorbar()
cbar.set_label('Kg m$^{-2}$ s$^{-1}$')
plt.gca().invert_yaxis()
plt.title('$^{14}$CO emissions from \'Nature\'')

#%% PLot the difference



final_14_1 = final / final_14

nintyperc = np.percentile(final_14_1,95)

pcolor(x,y,final_14_1,cmap = tmap, vmin = 0, vmax = nintyperc)
cbar = plt.colorbar()
cbar.set_label('Kg m$^{-2}$ s$^{-1}$')
plt.gca().invert_yaxis()







#%% Create netCDF4

import numpy as np
import netCDF4 as nc

ds = nc.Dataset('NAEIout.nc', 'w', format ='NETCDF4')

time = ds.createDimension('time',None)
lat = ds.createDimension('lat',812)
lon = ds.createDimension('lon',1378)

time = ds.createVariable('time','f4',('time',))
lats = ds.createVariable('lat','f4',('lat',))
lons = ds.createVariable('lon','f4',('lon',))

value = ds.createVariable('value','f4',('time','lat','lon',))
value.units = 'CO emmissions'

lats[:] = np.linspace(49.284432, 63, 812)
lons[:] = np.linspace(-8.1906224, 5 , 1378)
print('var size before adding data',value.shape)

value[0, :, :] = mapped_data

print('var size after adding 1st data',value.shape) 

#xval = np.linspace(1.0, 5.0, 1378)
#yval = np.linspace(1.0, 5.0, 812)
#value[1, :, :] = np.array(xval.reshape(-1,1) + yval)


print('var size after adding 2nd data',value.shape) 


ds.close()
#%%




data = nc.Dataset('NAEIout.nc','r')













#%%


point_sources = pd.read_excel('NAEIPointsSources_2018.xlsx','Data')
point_sources = point_sources[point_sources['Pollutant Name']=='CO']
point_sources.sort_values(['Emission'],inplace = True, ascending = False)
point_source_total = np.sum(point_sources['Emission'])

total_emissions = np.sum(sector_prod[:-2]) + point_source_total
sector_point_total = point_sources.groupby(['Sector']).sum() / 1000

#sector_point_total['Emission'] = 100 * sector_point_total['Emission'] / point_source_total




















