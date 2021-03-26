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


plot = True
D14_plot = False

system_dir = 'C:/Users/lpb20/OneDrive - Imperial College London/Documents/Odyssey/'
root_dir = system_dir + 'NAEI'


#%%
"Loading in the data and basic plots to begin exploring the data"


#Loading in the CO enhancemnt simulated as arriving at the site of interest seperated
#by SNAP sector

XCO = np.loadtxt(root_dir + '/COmixingratio.csv', delimiter = ',')
days_num = int(XCO.shape[0] / 24)



if plot:
    # Plotting the simulated enhancments by summing across all SNAP sectors
    tot_CO_enh = np.sum(XCO, axis = 1)
    sns.histplot(tot_CO_enh, binwidth=5)
    plt.xlabel('CO (ppb)')
    
    
    
    columns = ['Dom prod','energy prod','ind com','ind prod','nature','offshore','other trans',
               'road trans','solvent','waste']
    box_data = pd.DataFrame(XCO *100 / tot_CO_enh.reshape([24 * days_num,1]),columns = columns)
    plt.figure()
    box_data.boxplot()
    
    
    #Setting the national average emissions for each sector for use in plots
    columns_em_perc = np.array([29,5.5,13,7,2.5,0.05,27.7,13.5,0.12,2])
    #Plotting the % of emissions from each sector nationally
    for i,val in enumerate(columns_em_perc):
        plt.plot([i+0.5,i+1.5],[val,val],'r--')

    
    plt.ylabel('% of CO from sector')
    plt.xlabel('SNAP sector')
    
    
#%% Using approach from Rom et al
#Deciding if we want to calulate fall of values, or if we want to make nice plots
plot = False
#Excess CO calculations from txt file
XCO = np.loadtxt(root_dir + '/COmixingratio.csv', delimiter = ',')[:480]

#D14C values as calculated in Distribution calcs.xlsx
data = pd.read_excel(root_dir + '/Distribution calcs.xlsx','Sheet2',skiprows=1,index_col=0)
D14C = np.array(data.values).reshape([1,10])
Na = 6.02e23

conv_factor = (100 * 22400) / (1.169e-12 * Na * 1e-9)
#ie - pMC = conv_factor * (14C(mol cm-3) / CO (ppb))
#ie - D14C = (pmc/100 - 1) * 1000


# Loops for making plots for two different tree ages (60yrs = -250: 150 yrs = -327)
for s, tree_val in enumerate([-250, -327]):
    #Setting D14C value for Domestic Production based on wood age
    D14C[0][0] = tree_val
    #Converting D14C value to 14R (ratio 14C to 12C) values
    factor = ((1000 + D14C) *1e-3 * 1.176e-12)
    #Finding the 14CO enhancments based on calulations involving Co enhancments and 
    #D14C values for each sector
    CO14_conc = np.sum(((100 + (D14C / 10))/100) * 1.1694e-12  *XCO * 1e-9 * (1/24400) * Na, axis =1)
    
    #Setting the standard background values to start calculations from. This will effect
    #the simulated fall off rates
    CO_ppb_stand = 100
    CO_cm_3 = CO_ppb_stand *1e-9 * (1/22400) * 6.02e23
    D14_stand = 3000
    #Converting CO concentration and D14 to Co concentration in mol cm-3 using eq.1 
    #from Petrenko et al 2021
    CO14_cm_3_stand = 1.167e-12 * CO_cm_3 * ((D14_stand / 1000) + 1)
#    CO14_cm_3_stand = ((100 + (D14_stand / 10))/100) * 1.1694e-12  *CO_ppb_stand * 1e-9 * (1/24400) * Na 

    
    
    
    #Finding total CO simulated by adding enhancment to background values
    CO_ppb_pol = CO_ppb_stand + np.sum(XCO,axis = 1)
    #Cnverting from ppb to mol cm-3
    CO_cm_3_pol = CO_ppb_pol *1e-9 * (1/22400) * 6.02e23
    #Adding background 14CO values to enhancments calculated (both in mol cm-3)
    CO14_pol = CO14_cm_3_stand  + CO14_conc
    #Using total CO and 14CO simulated at site to find D14 expected there 
    pMC_pol = conv_factor * (CO14_pol / CO_ppb_pol)
    D14_pol = (pMC_pol - 100) * 10
    
    #Finding difference between simulated D14C and Background D14C
    DD14C = D14_pol - D14_stand
    #Defining excess Co to be plotting on x axis
    excess_CO = CO_ppb_pol - CO_ppb_stand
    error = 10 * (3181 * (0.25/CO_ppb_pol)+100 - 100)
    if s == 0:
        plt.errorbar(excess_CO, DD14C, error, fmt = 'x',label = 'young trees')
    else:
        plt.errorbar(excess_CO, DD14C, fmt = 'x', label = 'old trees')
    x = excess_CO
    if plot:
        x = np.linspace(0,200,200)
    
    
    #Calculating the change in D14C occuring from edition of pure fossil fuel
    CO_ppb_fossil = x + CO_ppb_stand
    CO_cm_3_fossil = CO_ppb_fossil *1e-9 * (1/22400) * 6.02e23
    CO14_cm_3_stand = 1.167e-12 * CO_cm_3 * ((D14_stand / 1000) + 1)
    CO14_fossil = CO14_cm_3_stand
    D14C_fossil = ((((CO14_fossil/CO_cm_3_fossil)/1.1694e-12) - 1) *1000) - D14_stand
    
    plt.plot(x,D14C_fossil,'r--')
    
    
    #Calulating change from addition of pure biogenic CO (with D14 = 60) 
    CO_ppb_bio = x + CO_ppb_stand
    CO_cm_3_bio = CO_ppb_bio *1e-9 * (1/22400) * 6.02e23
    D14_bio = (CO_ppb_stand * D14_stand + x * 106) / CO_ppb_bio
    CO14_cm_3_bio = 1.167e-12 * CO_cm_3_bio * ((D14_bio / 1000) + 1)
    
    D14C_bio = ((((CO14_cm_3_bio/CO_cm_3_bio)/1.1694e-12) - 1) *1000) - D14_stand
    
    plt.plot(x, D14C_bio, 'r--')
    
    plt.ylabel('$\Delta\Delta$$^{14}$C (per mil)')
    plt.xlabel('Excess CO (ppb)')

    # Analysing drop off rates
    for i in range(10):
        first_10_ppb = excess_CO[(excess_CO > 10 * i) & (excess_CO<10 * (i+1))]
        first_10_ppb_D14 = DD14C[(excess_CO > 10 * i) & (excess_CO<10 * (i+1))]
    
        print(str(10 * i)+' - '+ str(10 * (i+1)) + ':\t' + str(np.polyfit(first_10_ppb,first_10_ppb_D14,1)[0]))

    #Investiagting the Fossil Fractions modeling as simple two box model
    
    if s == 0:
        ff = pd.DataFrame((DD14C - D14C_bio) / (D14C_fossil - D14C_bio),columns = ['FF_new'])
    if s == 1:
        ff['FF_old'] = (DD14C - D14C_bio) / (D14C_fossil - D14C_bio)



plt.figure()
sns.boxplot(data = ff,)
plt.title('Fossil Fraction of simulations')

