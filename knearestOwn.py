# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 03:38:07 2017

@author: Eduardo RR
K nearest neightbors from scratch. Machine learning p 16
"""
#distancia en 3-nearest neighbors
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter
import warnings

style.use('fivethirtyeight')

dataset = {'k':[[1,2],[2,3],[3,1]],'r':[[6,5],[7,7],[8,6]]}
new_features = [5,7]


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
    print(Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0]
            
    return vote_result
#hacemos nuestra prediccion
result = k_nearest_neighbors(dataset,new_features,k=3)
print(result)

#ploteamos
[[plt.scatter(ii[0],ii[1],s=100,color=i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_features[0],new_features[1],s=100,color=result)
plt.show()  