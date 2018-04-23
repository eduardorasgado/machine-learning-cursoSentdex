# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 12:05:02 2017

@author: Orlando
"""

#R Squared Theory
#coeficiente de determinacion
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

def best_fit_slope_and_intercept(xy,ys):
    
    m = (((mean(xs)*mean(ys))-mean(xy*ys))/
         ((mean(xs)*mean(xs))-mean(xs*xs)))
    
    b = mean(ys)-m*mean(xs)
    return m,b

def squared_error(ys_orig,ys_line):
    return sum((ys_line-ys_orig)**2)

def coefficient_of_determination(ys_orig,ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig,ys_line)
    squared_error_y_mean = squared_error(ys_orig,y_mean_line)
    return 1 - (squared_error_regr/squared_error_y_mean)
    
    
xs = np.array([1,2,3,4,5,6],dtype=np.float64)
ys = np.array([5,4,6,5,6,7],dtype=np.float64)


m,b = best_fit_slope_and_intercept(xs,ys)

regression_line = [ ((m*x)+b) for x in xs]

r_squared = coefficient_of_determination(ys,regression_line)

print(r_squared)


plt.scatter(xs,ys,label='data')

plt.plot(xs, regression_line,label='regression line')
plt.legend(loc=4)
plt.show()
