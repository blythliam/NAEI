# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 08:14:05 2021

@author: lpb20
"""

#%%
"Importing the required libraries and parameters to determine what gets plotted"


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


plot = False
D14_plot = False

root_dir = 'C:/Users/lpb20/OneDrive - Imperial College London/Documents/Odyssey/NAEI'

#%%


#%%
"Loading in the data and basic plots to begin exploring the data"


#Loading in the CO enhancemnt simulated as arriving at the site of interest seperated
#by SNAP sector
#%% Using approach from Rom et al



XCO = np.loadtxt(root_dir + '/COmixingratio.csv', delimiter = ',')

data = pd.read_excel(root_dir + '/Distribution calcs.xlsx','Sheet2',skiprows=1,index_col=0)
D14C = np.array(data.values).reshape([1,10])

factor = ((1000 + D14C) *1e-3 * 1.176e-12)
CO14_conc = np.sum((0.989 * 2.6868e10 * factor  * XCO), axis =1)
CO_backgrounds = np.linspace(60,150,250)
D14_backgrounds = np.linspace(1000,5000,500)
sensitivity = np.zeros([len(D14_backgrounds), len(CO_backgrounds)])
    
for q, CO_ppb_stand in enumerate(CO_backgrounds):
    print(q)
    for j, D14_stand in enumerate(D14_backgrounds):
       
        
        CO_cm_3 = CO_ppb_stand *1e-9 * (1/22400) * 6.02e23
        
        
        CO14_cm_3_stand = 1.167e-12 * CO_cm_3 * ((D14_stand / 1000) + 1)
        
#        plt.plot(1e3/ppb_stand, (CO14_stand/ppb_stand), 'ro')
        
        
        CO_ppb_pol = CO_ppb_stand + np.sum(XCO,axis = 1)
        CO_cm_3_pol = CO_ppb_pol *1e-9 * (1/22400) * 6.02e23
        
        CO14_pol = CO14_cm_3_stand  + CO14_conc
        #CO14_pol_ppb = np.sum(XCO * factor, axis = 1)
        #pMC = 1/((1/(100 * CO14_pol)) * 1.169e-12 * 1e-9 * ppb_pol * (1/22400) * 6.02e23)
        
        D14_pol = ((CO14_pol/CO_cm_3_pol / 1.167e-12)-1) * 1000
        
        DD14C = D14_pol - D14_stand
#        sns.scatterplot(CO_ppb_pol - CO_ppb_stand, DD14C)
        excess_CO = CO_ppb_pol - CO_ppb_stand
        
        
        x = np.linspace(0,200,200)
        CO_ppb_fossil = x + CO_ppb_stand
        CO_cm_3_fossil = CO_ppb_fossil *1e-9 * (1/22400) * 6.02e23
        CO14_cm_3_stand = 1.167e-12 * CO_cm_3 * ((D14_stand / 1000) + 1)
        CO14_fossil = CO14_cm_3_stand
        D14C_fossil = ((((CO14_fossil/CO_cm_3_fossil)/1.1694e-12) - 1) *1000) - D14_stand
        
        
        
        
        CO_ppb_bio = x + CO_ppb_stand
        CO_cm_3_bio = CO_ppb_bio *1e-9 * (1/22400) * 6.02e23
        D14_bio = (CO_ppb_stand * D14_stand + x * 106) / CO_ppb_bio
        CO14_cm_3_bio = 1.167e-12 * CO_cm_3_bio * ((D14_bio / 1000) + 1)
        
        D14C_bio = ((((CO14_cm_3_bio/CO_cm_3_bio)/1.1694e-12) - 1) *1000) - D14_stand
        
        
        

        
        
        for i in range(1):
            first_10_ppb = excess_CO[(excess_CO > 10 * i) & (excess_CO<10 * (i+1))]
            first_10_ppb_D14 = DD14C[(excess_CO > 10 * i) & (excess_CO<10 * (i+1))]
            
#            print(np.polyfit(first_10_ppb,first_10_ppb_D14,1)[0])

        sensitivity[j,q] = np.polyfit(first_10_ppb,first_10_ppb_D14,1)[0]

#%%


x0 = CO_backgrounds[0]
x1 = CO_backgrounds[-1]
y0 = D14_backgrounds[0]
y1 = D14_backgrounds[-1]
plt.imshow(np.flipud(sensitivity), extent = [x0,x1,y0,y1],aspect='auto')
plt.colorbar()
#plt.plot(3000,110,'ro')
#plt.plot(7000,60,'bo')
plt.xlabel('Background CO (ppb)')
plt.ylabel('Background $\Delta^{14}$C')
sns.scatter([100],[3000])

#lt.figure()
#plt.contourf(sensitivity,levels = 10,origin = 'upper')



























