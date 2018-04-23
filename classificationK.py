# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 02:06:52 2017

@author: Orlando
"""

#classification intro with K nearest neighbors

"""crear un modelo que mejor divida o separe nuestros datos
Clustering: tienes un dataset y planeas dividorlo segun sus propiedades
k nearest: clasificar segun la distancia o cercania entre un punto y otros que estan definidos
por propiedades.
k es el numero de puntos que vamos a definir mas cercanos al punto a evaluar

k nearest neighbors es mejor que support vector machine
"""
#http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29

import numpy as np
from sklearn import preprocessing,cross_validation, neighbors
import pandas as pd
import sys
try:
    df = pd.read_csv('breast-cancer-wisconsin.data.txt')
    df.replace('?',-99999,inplace=True)
    
    #ahora buscamos los datos que no vamos a ocupar
    df.drop(['id'],1,inplace=True)
    #features: los features son todos menos la clase
    X = np.array(df.drop(['class'],1))
    
    #labels, el label a pronosticar es la clase de cancer
    y = np.array(df['class'])
    
    #entrenamos
    X_train, X_test, y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.2)
    
    #aplicamos el algoritmo
    clf = neighbors.KNeighborsClassifier()
    clf.fit(X_train,y_train)
    
    #medimos exactitud
    accuracy = clf.score(X_test,y_test) 
    
    print("la exactitud es de: ",accuracy)
    
    #ahora procedemos a hacer un pronostico
    example_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,2,2,2,3,2,1]]) #sin id y sin clase
    example_measures = example_measures.reshape(len(example_measures),-1)#el primero es el numero de arreglos arriba
    prediction = clf.predict(example_measures)
    
    print("La prediccion para los pacientes 1 y 2 son, en su clase: ",prediction)
except Exception as e:
    print(sys.exc_info()[0],sys.exc_info()[1],sys.exc_info()[2].tb_lineno,sep='\n')
    