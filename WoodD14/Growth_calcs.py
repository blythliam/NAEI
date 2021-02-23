# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 08:21:27 2021

@author: lpb20
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


hist_data = pd.read_csv('Historical_data.csv', skiprows=3, index_col=0).iloc[1:,:1]
future_data = pd.read_csv('Futureemissions.csv', encoding = 'Windows-1252',skiprows=3, index_col=0)
future_data = future_data.iloc[11:,:1]
hist_data.columns = future_data.columns

hist_data['D14C'] = hist_data['D14C'].astype(float)



tot_data = pd.concat([hist_data, future_data])
tot_data.index = np.floor(tot_data.index)

early_year = np.arange(0,tot_data.index[0],1)

early_values = np.repeat(tot_data.values[0],len(early_year))

early_years = pd.DataFrame(data = early_values, index = early_year)
early_years.columns = ['D14C']

tot_data = pd.concat([early_years,tot_data])


plt.plot(tot_data)

plt.xlim([1850,2020])


yrs = 100

onehundredyears = tot_data[(tot_data.index > 2020 - yrs) & (tot_data.index < 2020)]

years = np.array(onehundredyears.index)

vals = np.concatenate(onehundredyears.values, axis = 0)

Delta_14C = np.trapz(vals, years) / (years[-1] - years[0])

print('Trees that have been growing for '+str(yrs)+' yrs have a D14C of: ' + str(Delta_14C))
















