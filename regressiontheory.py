# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 10:18:06 2017

@author: Orlando
"""

#Regressiontheory and how it works
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

xs = np.array([1,2,3,4,5,6],dtype=np.float64)
ys = np.array([5,4,6,5,6,7],dtype=np.float64)


m,b = best_fit_slope_and_intercept(xs,ys)
print(m,b,sep="\n")

regression_line = [ ((m*x)+b) for x in xs]

predict_x = 8
predict_y = (m*predict_x)+b

plt.scatter(xs,ys,label='data')
plt.scatter(predict_x,predict_y,color='green')
plt.plot(xs, regression_line,label='regression line')
plt.legend(loc=4)
plt.show()