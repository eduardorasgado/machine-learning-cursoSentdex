# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 12:54:50 2017

@author: Orlando
"""

#Creating Sample Data for Testing p.12

#R Squared Theory
#coeficiente de determinacion
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

def create_dataset(how_many,variance,step=2,correlation=False):
    """Crea un conjunto de datos de ejemplo,aleatorios con una varianza,cantidad de datos
    y una correlacion determinadas, esto es crear un dataset de x y,con y aleatorios, pero
    con especifica correlacion y varianza"""
    val = 1
    ys = []
    for i in range(how_many):
        y =val + random.randrange(-variance,variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step
    xs = [i for i in range(len(ys))]
    
    return np.array(xs,dtype=np.float64), np.array(ys,dtype=np.float64)

def best_fit_slope_and_intercept(xy,ys):
    """Regresa el slop y el punto de intercepcion de la funcion mx+b"""
    m = (((mean(xs)*mean(ys))-mean(xy*ys))/
         ((mean(xs)*mean(xs))-mean(xs*xs)))
    
    b = mean(ys)-m*mean(xs)
    return m,b

def squared_error(ys_orig,ys_line):
    """regresa la sumatoria de las distancias entre los puntos de la grafica y
    la linea de regresion, esto es e**2"""
    return sum((ys_line-ys_orig)**2)

def coefficient_of_determination(ys_orig,ys_line):
    """Regresa el coeficiente de determinacion tomando la formula: 1 - SEy(hat)/SEy(media)
    siendo SE el error al cuadrado"""
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig,ys_line)
    squared_error_y_mean = squared_error(ys_orig,y_mean_line)
    return 1 - (squared_error_regr/squared_error_y_mean)
 
#creando dataset de ejemplo
#podemos cambiar estos datos
xs,ys = create_dataset(40,40,2,correlation= 'pos')
   
#xs = np.array([1,2,3,4,5,6],dtype=np.float64)
#ys = np.array([5,4,6,5,6,7],dtype=np.float64)

#extraccion de m y b, esto es pendiente e intercepcion
m,b = best_fit_slope_and_intercept(xs,ys)

#creando recta de regresion
regression_line = [ ((m*x)+b) for x in xs]

#haremos una prediccion con la regresion lineal que tenemos
predict_x = 50
predict_y = (m*predict_x)+b
            
#obteniendo el coeficiente del error, entre 0 y 1
r_squared = coefficient_of_determination(ys,regression_line)
print("el dataset es:",dict(zip(xs,ys)))
print("El coeficiente de determinacion es: ",r_squared)

plt.scatter(xs,ys,label='data')
plt.plot(xs, regression_line,label='regression line')
plt.scatter(predict_x,predict_y,color='green')
plt.legend(loc=4)
plt.show()