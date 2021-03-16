# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 11:22:11 2021

@author: lpb20
"""
##############################################################################
##########Code for exploring the data produced by 2kmanalysis.py##############
##############################################################################


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
"Loading in the data and basic plots to begin exploring the data"


#Loading in the CO enhancemnt simulated as arriving at the site of interest seperated
#by SNAP sector

XCO = np.loadtxt(root_dir + '/COmixingratio.csv', delimiter = ',')

days_num = XCO.shape[0]



if plot:
  
    sns.histplot(np.sum(XCO, axis = 1), binwidth=5)
    plt.xlabel('CO (ppb)')
    
    plt.figure()
    columns = ['Dom prod','energy prod','ind com','ind prod','nature','offshore','other trans',
               'road trans','solvent','waste']
    
    box_data = pd.DataFrame(XCO *100 / np.sum(XCO,axis = 1).reshape([24 * days_num,1]),columns = columns)
    box_data.boxplot()
    
    columns_em_perc = np.array([29,5.5,13,7,2.5,0.05,27.7,13.5,0.12,2])
    
    for i,val in enumerate(columns_em_perc):
        plt.plot([i+0.5,i+1.5],[val,val],'r--')

    
    plt.ylabel('% of CO from sector')
    plt.xlabel('SNAP sector')
    
#    plt.figure()
#    plt.plot(np.sqrt(C_of_m[:,0]**2 + C_of_m[:,1]**2), np.sum(XCO, axis = 1), 'x')
    
    
#    D14CO_pd = pd.DataFrame(np.zeros()
    
    if D14_plot:
        CO_median = []
        for i in np.linspace(-320,-320,1):
            D14 = np.array([i,-787,-787,-1000,33,-1000,-949,-949,-19,-449])
            
            factor = (1000 + D14)*1e-3 * 1.176e-12
            
            D14CO = 0.989 * 2.6868e10 * factor  * XCO
            D14CO_sum = np.sum(D14CO, axis = 1)
            CO_median.append(np.median(D14CO_sum))
            box_data = pd.DataFrame(D14CO / D14CO_sum.reshape([24 * days_num,1]),columns = columns)
            

#            plt.figure()
#            box_data.boxplot()
            plt.figure()
#            sns.histplot(D14CO_sum, stat = 'density')
            box_data.boxplot()
            plt.xlabel('$^{14}$CO (molecules cm$^{-3}$)')


    



#%% Using approach from Rom et al



XCO = np.loadtxt(root_dir + '/COmixingratio.csv', delimiter = ',')

data = pd.read_excel(root_dir + '/Distribution calcs.xlsx','Sheet2',skiprows=1,index_col=0)
D14C = np.array(data.values).reshape([1,10])

factor = ((1000 + D14C) *1e-3 * 1.176e-12)
CO14_conc = np.sum((0.989 * 2.6868e10 * factor  * XCO), axis =1)
import matplotlib.pyplot as plt

CO_ppb_stand = 100
CO_cm_3 = CO_ppb_stand *1e-9 * (1/22400) * 6.02e23
D14_stand = 3000

CO14_cm_3_stand = 1.167e-12 * CO_cm_3 * ((D14_stand / 1000) + 1)

#plt.plot(1e3/ppb_stand, (CO14_stand/ppb_stand), 'ro')


CO_ppb_pol = CO_ppb_stand + np.sum(XCO,axis = 1)
CO_cm_3_pol = CO_ppb_pol *1e-9 * (1/22400) * 6.02e23

CO14_pol = CO14_cm_3_stand  + CO14_conc
#CO14_pol_ppb = np.sum(XCO * factor, axis = 1)
#pMC = 1/((1/(100 * CO14_pol)) * 1.169e-12 * 1e-9 * ppb_pol * (1/22400) * 6.02e23)

D14_pol = ((CO14_pol/CO_cm_3_pol / 1.167e-12)-1) * 1000

DD14C = D14_pol - D14_stand
sns.scatterplot(CO_ppb_pol - CO_ppb_stand, DD14C)#,'bo',markersize = 0.1)
excess_CO = CO_ppb_pol - CO_ppb_stand


x = np.linspace(0,200,200)
CO_ppb_fossil = x + CO_ppb_stand
CO_cm_3_fossil = CO_ppb_fossil *1e-9 * (1/22400) * 6.02e23
CO14_cm_3_stand = 1.167e-12 * CO_cm_3 * ((D14_stand / 1000) + 1)
CO14_fossil = CO14_cm_3_stand
D14C_fossil = ((((CO14_fossil/CO_cm_3_fossil)/1.1694e-12) - 1) *1000) - D14_stand

plt.plot(x,D14C_fossil,'r--')



CO_ppb_bio = x + CO_ppb_stand
CO_cm_3_bio = CO_ppb_bio *1e-9 * (1/22400) * 6.02e23
D14_bio = (CO_ppb_stand * D14_stand + x * 106) / CO_ppb_bio
CO14_cm_3_bio = 1.167e-12 * CO_cm_3_bio * ((D14_bio / 1000) + 1)

D14C_bio = ((((CO14_cm_3_bio/CO_cm_3_bio)/1.1694e-12) - 1) *1000) - D14_stand

plt.plot(x, D14C_bio, 'r--')

plt.ylabel('$\Delta\Delta$$^{14}$C (per mil)')
plt.xlabel('Excess CO (ppb)')

#%% Analysing drop off rates

for i in range(20):
    first_10_ppb = excess_CO[(excess_CO > 10 * i) & (excess_CO<10 * (i+1))]
    first_10_ppb_D14 = DD14C[(excess_CO > 10 * i) & (excess_CO<10 * (i+1))]
    
    print(str(10 * i)+' - '+ str(10 * (i+1)) + ':\t' + str(np.polyfit(first_10_ppb,first_10_ppb_D14,1)[0]))

#%% Investiagting the Fossil Fractions


sns.scatterplot(excess_CO, DD14C)
plt.plot(x,D14C_fossil,'r--')
plt.plot(x, D14C_bio, 'r--')


ff = pd.DataFrame((DD14C - D14C_bio) / (D14C_fossil - D14C_bio),columns = ['FF'])
ff['Excess_CO'] = excess_CO
ff['Hours'] = (np.array(ff.index) % 24 < 6) | (np.array(ff.index) % 24 > 18)
ff['lat_cen'] = C_of_m[:,1]
ff['lon_cen'] = C_of_m[:,0]
ff['dist'] = np.sqrt((ff['lat_cen']-515)**2 + (ff['lon_cen']-110)**2)


plt.figure()
sns.boxplot(data = ff, x = 'FF')
plt.title('Fossil Fraction of simulations')

plt.figure()
sns.scatterplot(data = ff, x = 'Excess_CO', y = 'FF',hue = 'Hours')
plt.xlabel('Excess CO (ppb)')
plt.ylabel('Fossil Fuel Fraction')






#%%

data = pd.read_excel('Distribution calcs.xlsx','Sheet2',skiprows=1,index_col=0)
D14C = np.array(data.values).reshape([1,10])

factor = (1000 + D14C)*1e-3*1.176e-12
CO14_conc = np.sum(0.989 * 2.6868e10 * factor  * XCO, axis = 1)
bins = np.linspace(0,2,20)
plt.hist(CO14_conc, bins = bins)