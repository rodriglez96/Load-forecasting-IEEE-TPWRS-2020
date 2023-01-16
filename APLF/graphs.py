#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 15:48:39 2022

@author: rgonzalez
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
#archivo = 'results.csv'
archivo = 'results/results_aggregated.csv'
#archivo = 'results_mean.csv'
#archivo = 'results_1.csv'
datos = pd.read_csv(archivo)

datos['index'] = np.arange(len(datos))
timestamp = np.zeros((8760,1))
timestamp = pd.DataFrame(data.get('timestamp'))
timestamp = timestamp.rename(columns={0:'timestamp'})
timestamp = pd.to_datetime(timestamp['timestamp']-719529,unit='d').round('s')
datos = pd.concat([timestamp, datos], axis = 1)
error_min = datos.apply(lambda x: x['predictions']-x['estimated errors'], axis = 1)
error_max = datos.apply(lambda x: x['predictions']+x['estimated errors'], axis = 1)

fig, ax = plt.subplots() 
fig.set_size_inches(12, 8)
ax.plot(datos['timestamp'], datos['predictions'], color = 'blue')
ax.plot(datos['timestamp'], datos['load demand'], color = 'red')
ax.set_ylabel('Load demand')
ax.fill_between(datos['timestamp'], error_min, error_max, color = 'red', alpha = 0.1)
fig.legend(loc = 1)
fig.autofmt_xdate()
ax.set_xlim([datetime.date(2013, 1, 1), datetime.date(2013, 1, 7)])
fig

#fig2, ax = plt.subplots()
#fig2.set_size_inches(12, 8)
#ax.plot(consumption_m, color = 'blue')
#ax.plot(consumption_t, color = 'red')

