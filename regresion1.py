#MachineLearning
#Regresion

import pandas as pd
import quandl

#google stock in Quandl
df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume',]]
#HL Percent
df['HL_PCT']= (df['Adj. Open'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
#Percent change
df['PCT_change']= (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

print(df.head())

input()