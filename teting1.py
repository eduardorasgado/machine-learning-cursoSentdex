#testing1
#MACHINE LEARNING: es una serie de algoritmos que hacen que tu dispositivos o aplicacion tengan inteligencia
#Tensor flow: redes neuronales
#sklearn: logical core de machine learning

#Linear Regression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge

boston = load_boston()

print(str(boston.keys()),end="\n\n")

#print(boston.data)
print(boston.target.shape)

X_ent, X_test,y_ent, y_test = train_test_split(boston.data,boston.target)
print("ready...")
print(y_test.shape)

knn = KNeighborsRegressor(n_neighbors=3)

knn.fit(X_ent,y_ent)

print(knn.score(X_test,y_test))

input()
