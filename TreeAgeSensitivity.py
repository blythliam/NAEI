# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 15:39:44 2021

@author: lpb20
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


plot = False
D14_plot = False

root_dir = 'C:/Users/lpb20/OneDrive - Imperial College London/Documents/Odyssey/NAEI'

#%%



XCO = np.loadtxt(root_dir + '/COmixingratio.csv', delimiter = ',')
Decrease_rate=[]
ages = np.array([20,60])
D14C = np.zeros(len(ages))
for t, age in enumerate(ages):
    data = np.array(pd.read_excel(root_dir + '/Diff_Age_Trees/Distribution_calcs_'
                                  +str(age)+'.xlsx','Distribution calcs'
                                  ,index_col=None,header = None,skiprows=4,nrows=1))
    
    
    D14C[t] = data[0]
    
RCO = 1 + (D14C / 1000) 
   
        
RadioCO = RCO * 1.164e-12 * 0.989 * 10 * 2.6868e10
error = np.linspace(0.25, 0.25,len(RadioCO))

    
plt.errorbar(ages, RadioCO,error, fmt = 'rx',capsize = 4.5)
plt.ylim([0,0.75])
plt.title('Excess $^{14}$CO for addition of 10 ppb CO from trees of various ages')
plt.ylabel('excess $^{14}$CO (mol cm$^{-3}$ ppb)')
plt.xlabel('Age of tree')

#%%

XCO = np.loadtxt(root_dir + '/COmixingratio.csv', delimiter = ',')

D14_path = 'C:/Users/lpb20/OneDrive - Imperial College London/Documents/Odyssey/NAEI/Diff_Age_Trees'

for i in [20,60]:
    data = pd.read_excel(D14_path + '/Distribution_calcs_'+str(i)+'.xlsx','Sheet2',skiprows=1,index_col=0)
    D14C = np.array(data.values).reshape([1,10])
    
    factor = ((1000 + D14C) *1e-3 * 1.176e-12)
    CO14_conc = np.sum((0.989 * 2.6868e10 * factor  * XCO), axis =1)
    
    CO_ppb_stand = 100
    D14_stand = 2500
    
           
            
    CO_cm_3 = CO_ppb_stand *1e-9 * (1/22400) * 6.02e23
            
            
    CO14_cm_3_stand = 1.167e-12 * CO_cm_3 * ((D14_stand / 1000) + 1)
            
    #plt.plot(1e3/ppb_stand, (CO14_stand/ppb_stand), 'ro')
           
            
    CO_ppb_pol = CO_ppb_stand + np.sum(XCO,axis = 1)
    CO_cm_3_pol = CO_ppb_pol *1e-9 * (1/22400) * 6.02e23
            
    CO14_pol = CO14_cm_3_stand  + CO14_conc
          #CO14_pol_ppb = np.sum(XCO * factor, axis = 1)
         #pMC = 1/((1/(100 * CO14_pol)) * 1.169e-12 * 1e-9 * ppb_pol * (1/22400) * 6.02e23)
          
    D14_pol = ((CO14_pol/CO_cm_3_pol / 1.167e-12)-1) * 1000
            
    DD14C = D14_pol - D14_stand
#    sns.scatterplot(CO_ppb_pol - CO_ppb_stand, DD14C)
    plt.plot(CO_ppb_pol - CO_ppb_stand, DD14C,'x')
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
                
    print(np.polyfit(first_10_ppb,first_10_ppb_D14,1)[0])
    
    #        sensitivity[j,q] = np.polyfit(first_10_ppb,first_10_ppb_D14,1)[0]
    




    
    
    