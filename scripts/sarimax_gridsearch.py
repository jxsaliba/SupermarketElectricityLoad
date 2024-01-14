# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 11:20:41 2021

@author: Jonathan Saliba

This will aim to introduce grid search iteratively to
find the best fitting parameters p, d and q and P, D, Q and S
for SARIMAX
"""

import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
import warnings
import itertools
import statsmodels.api as sm
import time
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_percentage_error as mape
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import copy

path_scripts = os.getcwd()
path_parent = os.path.dirname(path_scripts)
path_inputs= path_parent + '\Inputs'
path_outputs= path_parent + '\Outputs'

warnings.filterwarnings("ignore")

first_day_training = '2020-10-01'
first_day_testing = '2020-10-06'


os.chdir(path_inputs)
#A is the original sunningdate data 
A = pd.read_csv('sunningdale_fortesting.csv')
A=A.dropna()
def getweekday(x):
    import datetime
    y = datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S').weekday() 
    
    return y


def getdayofyear(x):
    import pandas as pd
    y = pd.Period(x,freq='D').day_of_year
    return y
    

A['weekday'] = A['Date_object'].apply(getweekday)
A['isSunday'] = A['weekday'].apply(lambda x:  1 if x == 6 else 0)
A['day_of_year'] = A['Date_object'].apply(getdayofyear)



#B is the Sunday Series, C is the Monday to Saturday series
B = A.where(A['isSunday'] ==1).dropna()
C = A.where(A['isSunday']==0).dropna()




test_no_of_timesteps = 48



index_firstday_training = A.loc[A['Date_object']== first_day_training + ' 00:00:00'].index[0]
index_lastday_training = A.loc[A['Date_object']== first_day_testing + ' 00:00:00'].index[0]


train = A.iloc[index_firstday_training:index_lastday_training,:]
test =A.iloc[index_lastday_training: index_lastday_training+ test_no_of_timesteps,:]

test_kWh = test['kWh']
train_series  = train['kWh']
test_series = test['kWh']
weekday_series_train = train['weekday']
temperature_series_train = train['Temperature']
sunday_series_train = train['isSunday']

weekday_series_test = test['weekday']
temperature_series_test = test['Temperature']
sunday_series_test= test['isSunday']

exog_train = np.column_stack((weekday_series_train.to_numpy(),temperature_series_train.to_numpy()))
exog_test = np.column_stack((weekday_series_test.to_numpy(),temperature_series_test.to_numpy()))



# gridsearch_results = pd.DataFrame(columns=['parameters','AIC'])


# Define the p, d and q parameters to take any value between 0 and 2
#p = d = q = range(0, 2)
# p = q = range(2,8)
# d = range(0,2)

# p = q = [2, 3]
# d = [0, 1]

# Generate all different combinations of p, q and q triplets
#pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, q and q triplets
#seasonal_pdq = [(x[0], x[1], x[2], 48) for x in list(itertools.product(p, d, q))]

# print('Examples of parameter combinations for Seasonal ARIMA...')
# print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
# print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
# print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
# print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))


# warnings.filterwarnings("ignore") # specify to ignore warning messages
# counter = -1
tic = time.perf_counter()
# for param in pdq:
#     for param_seasonal in seasonal_pdq:
#         counter = counter +1
#         try:
#             mod = sm.tsa.statespace.SARIMAX(train_series,
#                                             order=param,
#                                             seasonal_order=param_seasonal,
#                                             enforce_stationarity=False,
#                                             enforce_invertibility=False)

#             results = mod.fit()

#             print('SARIMAX{}x{}48 - AIC:{}'.format(param, param_seasonal, results.aic))
#             paramstore = str(param)+' '+str(param_seasonal)
#             gridsearch_results['parameters'].iloc[counter]= paramstore
#             gridsearch_results['AIC'].iloc[counter]=results.aic
            
#         except:
#             continue

param = (1,0,1)#increassing p does help a bit , but not much
param_seasonal= (0,1,1,48)


mod = sm.tsa.statespace.SARIMAX(train_series,
                                exog= temperature_series_train,
                                order=param,                                
                                seasonal_order=param_seasonal,
                                enforce_stationarity=False,
                                enforce_invertability=False)
mod_fit = mod.fit()

plot_acf(train_series,lags=25)
plt.show()
plot_pacf(train_series,lags=25)
plt.show()
        

print('First day of training: ', first_day_training)
print('First(only) day of testing: ',first_day_testing)

adfuller_result  = adfuller(train_series)
print('ADF Statistic: %f' % adfuller_result[0])
print('p-value: %f' % adfuller_result[1])
# p-value is 0.000012
# so the data is stainionary because it is < 0.05

toc = time.perf_counter()
timetaken = toc-tic
print('time in sec taken to run: ',timetaken)
print('AIC: ', mod_fit.aic)

output = mod_fit.forecast(steps=48,exog=temperature_series_test)

x  = np.linspace(1,len(test_series),len(test_series))
plt.plot(x,output,color='red',label='Predicted kWh')
plt.plot(x,test_kWh,color='blue',label='Actual kWh')
plt.legend()


MAPE = mape(test_series,output)
MSE = mse(test_series,output)

print('MAPE: ', MAPE)
print('MSE: ',MSE)



