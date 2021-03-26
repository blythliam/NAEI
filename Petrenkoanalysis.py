# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 13:34:52 2021

@author: lpb20
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from datetime import datetime

data_path = 'C:/Users/lpb20/OneDrive - Imperial College London/Documents/Odyssey/NAEI/Petrenko.csv'

data = pd.read_csv(data_path, skiprows=1, index_col=None,parse_dates = [0])[:36]

months = []
summer = []
for date in data['Date ']:
    dt = datetime.strptime(date, '%d/%m/%Y')
    months.append(dt.month)
    if dt.month >= 6 and dt.month <= 11:
        summer.append('Yes')
    else:
        summer.append('No')
data['month'] = months
data['month'] = np.floor(data['month']/3)
data['summer'] = summer

data['error'] = 3181.92 * (data['14CO'] / data['CO'])
averages = data.groupby(['summer']).mean()

#re.search(, string)

data_summer = data[data['summer'] == 'Yes']
data_winter = data[data['summer'] == 'No']


#plt.errorbar(data_summer['CO'], data_summer['D14C'], data_summer['error'],fmt = 'x')
#plt.errorbar(data_winter['CO'], data_winter['D14C'], data_winter['error'],fmt = 'x')


sns.scatterplot(data = data, x = 'CO', y = 'D14C',hue = 'month')

plt.xlabel('CO (ppb)')
plt.ylabel('$\Delta^{14}$C (per mil)')

plt.ylim([0,3500])
plt.xlim([0,150])



for i in range(5):
    data_new = data[data['month'] == i]
    print(data_new)
    if len(data_new)>0:
        m = np.polyfit(data_new['CO'],data_new['D14C'],1)[0]
        c = np.polyfit(data_new['CO'],data_new['D14C'],1)[1]
        x = np.linspace(np.min(data_new['CO']), np.max(data_new['CO']))
        y = m * x + c
        plt.plot(x,y)


plt.plot(averages['CO'], averages['D14C'],'rx')





