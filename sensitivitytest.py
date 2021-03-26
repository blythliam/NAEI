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



#%% Using approach from Rom et al



XCO = np.loadtxt(root_dir + '/COmixingratio.csv', delimiter = ',')

data = pd.read_excel(root_dir + '/Distribution calcs.xlsx','Sheet2',skiprows=1,index_col=0)
D14C = np.array(data.values).reshape([1,10])

factor = ((1000 + D14C) *1e-3 * 1.176e-12)
CO14_conc = np.sum((0.989 * 2.6868e10 * factor  * XCO), axis =1)
CO_backgrounds = np.linspace(60,250,250)
D14_backgrounds = np.linspace(1000,6000,500)
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
cbar = plt.colorbar()
cbar.set_label('$\Delta\Delta^{14}$C (per mille ppb$^{-1}$)', rotation=270,labelpad = 15)
#plt.plot(3000,110,'ro')
#plt.plot(7000,60,'bo')
plt.xlabel('Background CO (ppb)')
plt.ylabel('Background $\Delta^{14}$C (per mille)')


CO_nothern = {}

CO_nothern['NORWAY_SUMMER'] = [77.5,2917,'summer','NORWAY']
CO_nothern['NORWAY_WINTER'] = [170,3492,'winter','NORWAY']
CO_nothern['CANADA_SUMMER'] = [87.5,2602,'summer','CANADA']
CO_nothern['CANADA_WINTER'] = [170,3445,'winter','CANADA']
CO_nothern['WASHINGTON_SUMMER'] = [108,1579,'summer','WASHINGTON']
CO_nothern['WASHINGTON_WINTER'] = [195,2371,'winter','WASHINGTON']
CO_nothern['ALASKA_SUMMER'] = [107.5,2033,'summer','ALASKA']
CO_nothern['ALASKA_WINTER'] = [195,3103,'winter','ALASKA']
#CO_nothern['HAWAII_SUMMER'] = [79,1979,'summer','HAWAII']
#CO_nothern['HAWAII_WINTER'] = [90,2315,'winter','HAWAII']


CO_nothern['WASHINGTON_SUMMER1'] = [100,1545,'summer','WASHINGTON']
CO_nothern['WASHINGTON_SUMMER2'] = [90,1474,'summer','WASHINGTON']
CO_nothern['WASHINGTON_SUMMER3'] = [90,2181,'summer','WASHINGTON']
CO_nothern['WASHINGTON_SUMMER4'] = [140,1045,'summer','WASHINGTON']
CO_nothern['WASHINGTON_SUMMER5'] = [120,1651,'summer','WASHINGTON']

CO_nothern['WASHINGTON_WINTER1'] = [210,1727,'winter','WASHINGTON']
CO_nothern['WASHINGTON_WINTER2'] = [200,2659,'winter','WASHINGTON']
CO_nothern['WASHINGTON_WINTER3'] = [220,2363,'winter','WASHINGTON']
CO_nothern['WASHINGTON_WINTER4'] = [200,2181,'winter','WASHINGTON']
CO_nothern['WASHINGTON_WINTER5'] = [180,2358,'winter','WASHINGTON']
CO_nothern['WASHINGTON_WINTER6'] = [160,2977,'winter','WASHINGTON']

#CO_nothern['JAPAN_SUMMER'] = [149.5,1488,'summer','JAPAN']
#CO_nothern['JAPAN_WINTER'] = [197,2145,'winter','JAPAN']
#CO_nothern['BARBADOS_SUMMER'] = [82,1384,'summer','BARBADOS']
#CO_nothern['BARBADOS_WINTER'] = [98,1857,'winter','BARBADOS']
#CO_nothern['SWITZERLAND_SUMMER'] = [106,2110,'summer','SWITZERLAND']
#CO_nothern['SWITZERLAND_WINTER'] = [141.25,2370,'winter','SWITZERLAND']




CO_northernhemisphere = pd.DataFrame.from_dict(CO_nothern,orient = 'index', columns = ['CO','D14C','season','LOC'])
CO_WASHINTON = CO_northernhemisphere[CO_northernhemisphere['LOC'] == 'WASHINGTON']

#sns.scatterplot(data = CO_WASHINTON, x = 'CO',y = 'D14C', hue = 'season')
sns.scatterplot(data = CO_northernhemisphere, x = 'CO',y = 'D14C',hue = 'LOC')

# plt.ylim([0,4500])
# plt.xlim([0,350])


#%%

CO_northern_summer = CO_northernhemisphere[CO_northernhemisphere['season'] == 'summer']

m = np.polyfit(CO_northern_summer['CO'],CO_northern_summer['D14C'],1)[0]
c = np.polyfit(CO_northern_summer['CO'],CO_northern_summer['D14C'],1)[1]

x = np.linspace(np.min(CO_northern_summer['CO'])-10, np.max(CO_northern_summer['CO']))
y = m * x + c
plt.plot(x,y)
plt.plot(80, m * 80 + c, 'ko')

CO_northern_winter = CO_northernhemisphere[CO_northernhemisphere['season'] == 'winter']

m = np.polyfit(CO_northern_winter['CO'],CO_northern_winter['D14C'],1)[0]
c = np.polyfit(CO_northern_winter['CO'],CO_northern_winter['D14C'],1)[1]

x = np.linspace(140, 225)
y = m * x + c
plt.plot(x,y)

plt.plot(140, m * 140 + c, 'ko')





#%%

XCO = np.loadtxt(root_dir + '/COmixingratio.csv', delimiter = ',')

data = pd.read_excel(root_dir + '/Distribution calcs.xlsx','Sheet2',skiprows=1,index_col=0)
D14 = np.array(data.values).reshape([1,10])

RCO = 1 + (D14/1000)

CO_enh = np.sum(XCO, axis = 1)
CO14_enh = np.sum(RCO * 1.164e-12 * 0.989 * XCO * 2.6868e10, axis = 1)






Bioenh = np.linspace(0,200,201)
    
RCO = 1 + (200/1000)

RadioCO = RCO * 1.164e-12 * 0.989 * Bioenh * 2.6868e10

error = np.linspace(0.25, 0.25,len(RadioCO))

plt.errorbar(Bioenh, RadioCO, yerr = error, fmt = 'gx', capsize = 4.5)
plt.errorbar(Bioenh,np.zeros(len(Bioenh)), error, fmt = 'kx',capsize=4.5)

plt.xlabel('CO enhancments (ppb)')
plt.ylabel('$^{14}$CO enhancments (mol cm$^{-3}$)')


error = np.linspace(0.25,0.25,len(CO_enh))
plt.errorbar(CO_enh, CO14_enh,error, fmt ='x', capsize = 4.5)

plt.legend(['Biogenic CO','Fossil CO','Simulated CO'])

plt.plot([min(CO_enh),max(CO_enh)],[min(CO14_enh),max(CO14_enh)],'b',zorder = 10)






















