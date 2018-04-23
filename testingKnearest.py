# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 04:42:14 2017

@author:Eduardo RR
Testing our K-nearest Neighbours classifier

Este algoritmo puede ser usado sobre lineares y no lineares datos
"""
import numpy as np
from math import sqrt
from collections import Counter
import warnings
import pandas as pd
import random


def k_nearest_neighbors(data,predict,k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')
    #knnalgos
    distances = []
    for group in data:
        for features in data[group]:
            #euclidean_distance = sqrt((features[0]-predict[0])**2 + (features[1]-predict[1])**2)
            #lo anterior no es muy rapido asi que lo mejoramos:
            #euclidean_distance = np.sqrt(np.sum((np.array(features)-np.array(predict))**2))
            #mejor aun con liibreria especializada de numpy:
            #con esta forma nos aseguramos de las dimensiones que sean
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance,group])
    votes = [i[1] for i in sorted(distances)[:k]]
    #print(Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1]/k
        
    return vote_result, confidence

accuracies = []
for num in range(5):
    #abrimos nuestros datos
    df = pd.read_csv('breast-cancer-wisconsin.data.txt')
    df.replace('?',-99999,inplace=True)
        
    #ahora buscamos los datos que no vamos a ocupar
    df.drop(['id'],1,inplace=True)
    
    #nos aseguramos que todos los datos se conviertan en flotantes, ya que algunos
    #de ellos son de alguna forma strings o estan entre comillas
    full_data = df.astype(float).values.tolist()
    #print(full_data[:5])
    #para ver que estan entre comillas,intentar:
    #try_data = df.values.tolist()
    #print(try_data[:5]
    #print("El quinto elemento del primer miembro es del tipo: ",str(type(try_data[0][5]))[1:12],sep='' )
    print(15*'-')
    
    #desordenamos los datos
    random.shuffle(full_data)
    #random_shuffle cambia el dataset realmente, sin reasignarlo a otra var   
    #print(full_data[:5])
    
    test_size = 0.2
    train_set = {2:[],4:[]}
    test_set = {2:[],4:[]}
    
    #todos los daatos menos los ultimos 20% de ellos
    train_data = full_data[:-int(test_size*len(full_data))]
    
    #El 20 porciento que se quitó del dato de entrenamiento 
    test_data = full_data[-int(test_size*len(full_data)):] #los ultimos 20%
    
    #agregando a los tes y train sets
    for i in train_data:
        #la columna class:2 o 4 agregarle todos los valores q no sean de clase o id
        train_set[i[-1]].append(i[:-1])
    
    for i in test_data:
        #la columna class:2 o 4 agregarle todos los valores q no sean de clase o id
        test_set[i[-1]].append(i[:-1])
        
    correct = 0
    total = 0
    for group in test_set:
        for data in test_set[group]:
            vote, confidence = k_nearest_neighbors(train_set,data,k=5)
            if group == vote:
                correct += 1
            else:
                print("incorrect testing,confidence is: ",confidence)
            total +=1
        
    print("accuracy for testing #{} is: ".format(num+1),correct/total)
    accuracies.append(correct/total)

print("la exactitud de las pruebas es de: ",sum(accuracies)/len(accuracies))
    
