# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 05:15:04 2017

@author: Orlando
"""

#definiendo etiquetas:labels
"""COmenzamos definiendo etqieutas para hacer predicciones  traves de la recursion"""

#MachineLearning
#Regresion

import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing, cross_validation,svm
from sklearn.linear_model import LinearRegression
"""preprocessing es usado para hacer algun limpiado/escalado de los datos ocupados 
en machine learning
cross_validation es usado principalmente para testear stages 
"""
""" con el supervised learning conseguimos labels y features, los features son
atributos descriptivos y los labels son aquello que intentamos predecir
forecast significa pronostico
svm:support vector machine"""

#google stock in Quandl
df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume',]]
#HL Percent
df['HL_PCT']= (df['Adj. Open'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
#Percent change
df['PCT_change']= (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

#trabajando con el agoritmo de regresion
forecast_col = 'Adj. Close'
df.fillna(value=-99999,inplace=True)  #just in case lose data  o un valor NaN

forecast_out = int(math.ceil(0.01*len(df))) #lo que equivale a 1% del total de dias 
#de la estancia de este stock por tanto nuestros datos son 100 y nuestro 1% es 0.001
#labels
df['label']= df[forecast_col].shift(-forecast_out)

print(df.head())
