# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 14:05:11 2021

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






NH_data = [float(i) for i in hist_data['D14C']]
#tropic_data = [float(i) for i in hist_data['Tropics Delta14co2']]
#SH_data = [float(i) for i in hist_data['SH Delta14co2']]

#plt.plot(list(hist_data.index), NH_data)
#plt.plot(list(hist_data.index), tropic_data)
#plt.plot(list(hist_data.index), SH_data)
#plt.plot(future_data.index, future_data['D14C'],'kx')


from scipy.integrate import odeint
def model(Rbio, t, RA, tau):
    dRbiodt = (1/tau) * (RA - Rbio) - (1/8267) * Rbio
    return dRbiodt


t = tot_data.index
tau = 100
RA = tot_data.values
Rbio0 = 0 
Rbios = np.zeros([len(RA),4])
Rbios[0] = Rbio0
m = 50
for j in range(0,4):

    tau = (j+1) * m
    for i in range(len(t) - 1):
        ts = [t[i],t[i+1]]
        NA = odeint(model, Rbio0, ts, args = (RA[i][0],tau))
        Rbio0 = NA[1]
        Rbios[i+1,j] = Rbio0



plt.plot(tot_data.index, tot_data.values)
plt.xlim([1850,2020])
Rbios_df = pd.DataFrame(data = Rbios, index = tot_data.index)
Rbios_df.columns = list((np.arange(0,4)+1) * m)
plt.plot(Rbios_df.index, Rbios_df.values)
plt.legend(['RA','t = '+str(Rbios_df.columns[0])+' yrs','t = '+str(Rbios_df.columns[1])+' yrs','t = '+str(Rbios_df.columns[2])+' yrs','t = '+str(Rbios_df.columns[3])+' yrs'])
plt.xlabel('Year')
plt.ylabel('$\Delta$$^{14}$C')
for i,column in enumerate(Rbios_df):
    print('For a turnover time of '+str(Rbios_df.columns[i]) + ' years, the final value is ' + str(Rbios_df.iloc[2020,i]))

























