# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 05:53:42 2017

@author: Orlando
"""

#training and t4sting
import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing, cross_validation,svm
from sklearn.linear_model import LinearRegression

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
print("forecast out:",forecast_out)
#labels
df['label']= df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

X = np.array(df.drop(['label'],1))
y = np.array(df['label'])
X = preprocessing.scale(X)
y = np.array(df['label'])

#training and testings
#Shuffle them, keeping x, y conected, it shoves them pu
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)
#classifier
clf = LinearRegression(n_jobs=-1)  #podemos usar adentro: n_jobs=1, que nos dice cuantos threads podemos usar
clf.fit(X_train,y_train)
accuracy = clf.score(X_test,y_test)

print("accuracy by LR:",accuracy)

#Usando distintas formas de la arquitectura svm
for k in ['linear','poly','rbf','sigmoid']:  #podemos tomar como el efectivo a linear
    clf = svm.SVR(kernel=k)  #support vector machine es como se ve, menos exacto
    clf.fit(X_train,y_train)
    accuracy = clf.score(X_test,y_test)
    
    print("accuracy:",k,accuracy)
    

