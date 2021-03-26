# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 11:35:19 2021

@author: lpb20
"""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#%%
"Look at CO enhancements from the different source sectors over historical period"

root_dir = 'C:/Users/lpb20/OneDrive - Imperial College London/Documents/Odyssey/NAEI'

XCO = np.loadtxt(root_dir + '/COmixingratio.csv', delimiter = ',')
days_num = XCO.shape[0]
cols = np.arange(3,15)

conversion_factor = pd.read_excel(root_dir + '/AllYearCO.xlsx',sheet_name='SNAP_Sectors',skiprows=2,index_col=0,usecols=cols).iloc[:-2,:]

normalise_fact = np.array(conversion_factor.iloc[:,-1])
                          


normalised_conversion_factor = conversion_factor.divide(normalise_fact,axis = 0)

medXCO = np.median(XCO, axis = 0)

XCOovertime = normalised_conversion_factor.multiply(medXCO,axis = 0).transpose()

XCOovertime.plot(kind = 'bar', stacked = True)

plt.ylabel('Median CO enhancment (ppb)')


#%%
"Looking at how CO changes in Domestic Combustion"

Domestic_comb = pd.read_excel(root_dir + '/AllYearCO.xlsx',sheet_name='Select_Dom',skiprows=0,index_col=0)

Domcombovertime = Domestic_comb.transpose()

Domcombovertime.plot(kind = 'bar', stacked = True)

plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

#%%

CO1970 = []
CO1975 = []
CO1980 = []
CO1985 = []
CO1990 = []
CO1995 = []
CO2000 = []
CO2005 = []
CO2010 = []
CO2015 = []
CO2018 = []

hist_data = pd.DataFrame(np.zeros([24984,11]),columns=normalised_conversion_factor.columns)

for XCO_sep in XCO:
    tot = normalised_conversion_factor.multiply(XCO_sep,axis = 0)
    totCO = tot.sum()
    CO1970.append(totCO[1970])    
    CO1975.append(totCO[1975]) 
    CO1980.append(totCO[1980])
    CO1985.append(totCO[1985]) 
    CO1990.append(totCO[1990]) 
    CO1995.append(totCO[1995]) 
    CO2000.append(totCO[2000]) 
    CO2005.append(totCO[2005]) 
    CO2010.append(totCO[2010]) 
    CO2015.append(totCO[2015]) 
    CO2018.append(totCO[2018]) 






hist_data[1970] = CO1970
hist_data[1975] = CO1975
hist_data[1980] = CO1980
hist_data[1985] = CO1985
hist_data[1990] = CO1990
hist_data[1995] = CO1995
hist_data[2000] = CO2000
hist_data[2005] = CO2005
hist_data[2010] = CO2010
hist_data[2015] = CO2015
hist_data[2018] = CO2018


med_vals = hist_data.median()
max_values = hist_data.median() + hist_data.mad()

for year in hist_data.columns:
    hist_data[year].mask(hist_data[year]>max_values[year],np.nan,inplace = True)


hist_data.boxplot()
#plt.ylim([0,1000])

plt.ylabel('CO enhancments (ppb)')
plt.xlabel('Year')






























