# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 08:46:32 2021

@author: Jonathan Saliba
"""

from pandas import read_csv
from statsmodels.tsa.arima_model import ARIMA
import numpy
from matplotlib import pyplot as plt
from pandas import datetime
from sklearn.metrics import mean_squared_error as mse
import os
import pandas as pd
import time
from sklearn.metrics import mean_absolute_percentage_error as MAPE
import copy
import metrics
import datetime as dt


class SunningdaleSweep:
    pass



path_scripts = os.getcwd()
path_parent = os.path.dirname(path_scripts)
path_inputs= path_parent + '\Inputs'
path_outputs= path_parent + '\Outputs'

def getdayofyear(x):
    from datetime import datetime
    #y = datetime.strptime(x,'%Y-%m-%d %H:%M').timetuple().tm_yday
    y = datetime.strptime(x, '%d/%m/%Y %H:%M').timetuple().tm_yday
    return y

os.chdir(path_inputs)
merged_df = pd.read_csv('sunningdale_merged.csv')
merged_df = merged_df.drop(columns=['Year_x','Month_x'])
merged_df = merged_df.rename(columns={"Year_y": "Year", "Month_y": "Month"})
merged_df['Day_of_year'] = merged_df['Date'].apply(getdayofyear)

def getdatetime(x):
    import datetime as dt
    day = int(x[0:2])
    month = int(x[3:5])
    year = int(x[6:10])
    hour = int(x[11:13])
    
    minute = int(x[14:18]) 
    y = dt.datetime(year, month,day,hour, minute)
    return y

merged_df['Date_object'] = merged_df['Date'].apply(getdatetime)

mask_2020 = merged_df['Year'] == 2020

merged_df2020 = merged_df.loc[mask_2020]
merged_df2020 =merged_df2020.set_index(pd.DatetimeIndex(merged_df2020['Date_object']))
merged_df2020 = merged_df2020.resample('30min').mean()
merged_df2020 = merged_df2020.interpolate(method='linear', limit_direction='forward', axis=0)

mask_2021 = merged_df['Year'] == 2021

merged_df2021 = merged_df.loc[mask_2021]
merged_df2021 =merged_df2021.set_index(pd.DatetimeIndex(merged_df2021['Date_object']))

merged_df2020 = merged_df2020[['kWh','Temperature']]
merged_df2021 = merged_df2021[['kWh','Temperature']]

df = pd.concat([merged_df2020,merged_df2021])
df = df.reset_index()
df = df.reset_index()
df = df.set_index(pd.DatetimeIndex(df['Date_object']))

gap = pd.read_csv('sunningdale_loadgap2021.csv')
gap = gap.set_index(pd.DatetimeIndex(gap['Date']))
concatenate = pd.concat([df,gap])
concatenate= concatenate.sort_index()

df = copy.deepcopy(concatenate)

df = df[['kWh','Temperature']]
df = df.reset_index()
df = df.set_index(pd.DatetimeIndex(df['index']))
df.rename(columns={'index':'Date_object'}, inplace=True)

df.to_csv('sunningdale_fortesting.csv')

rerun_periods =[1]# [7,14,30,60]
sample_sizes = [30]#[30,60,90,120]
number_of_predicted_days = 1

prediction_date_start = dt.datetime(2021,5,26,0,0)
prediction_date_start_str = prediction_date_start.strftime('%Y-%m-%d')

prediction_date_end = prediction_date_start + dt.timedelta(days=1)
prediction_date_end_str = prediction_date_end.strftime('%Y-%m-%d')

for rerun_period in rerun_periods:
    for sample_size in sample_sizes:


        pds = prediction_date_start
        train_end = prediction_date_start #- dt.timedelta(days=rerun_period)
        train_end_str = train_end.strftime('%Y-%m-%d')
        train_start = train_end - dt.timedelta(days=sample_size)
        train_start_str = train_start.strftime('%Y-%m-%d')
        
        test_end_str = prediction_date_end_str
        test_start_str = train_end_str
                
        print('Sample Size:', sample_size)
        print('Rerun period:', rerun_period)
        print('Prediction date:', prediction_date_start_str)
        print('Sample/Train start date:', train_start_str )
        print('Sample/Train End Date:', train_end_str)
        # pred = 1
        # name_input = 'Rerun'+str(rerun_period)+'Sample'+str(sample_size)
        # exec("%s = %d" % (name_input,rerun_period))
        # x = copy.deepcopy(globals()[name_input])       
        # sweep.x = globals()[name_input] 
        
        # test_start_str= train_end_str #end of training time is equivalent to startof test date range
        
        mask_train = (df['Date_object'] >= train_start_str) & (df['Date_object'] < train_end_str)
        mask_test = (df['Date_object'] >= test_start_str) & (df['Date_object'] < test_end_str)        
        
        train = df.loc[mask_train]
        test =  df.loc[mask_test]        
        
        train_kWh = train['kWh'].to_numpy()
        test_kWh = test['kWh'].to_numpy()
        forecasts = list()       
        
        history = [x for x in train_kWh]
        model = ARIMA(history, order=(7,0,1))
        model_fit = model.fit(maxiter=10,solver='cg')
        output = model_fit.forecast(steps=48)[0]
        
        predicted_kWh = numpy.array(output)
        #predicted_kWh = numpy.array([Z[0] for Z in predicted_kWh])
        
        test_kWh_df = pd.DataFrame(test_kWh)
        predicted_kWh_df = pd.DataFrame(predicted_kWh)
        
        
x  = numpy.linspace(1,len(test_kWh),len(test_kWh))

plt.plot(x,predicted_kWh,color='red',label='Target kWh')
plt.plot(x,test_kWh,color='blue',label='Test kWh')
plt.show()
plt.legend()
