#Linear classification
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn import datasets
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use("fivethirtyeight")

#importing the flower dataset
iris = datasets.load_iris()
X_iris, y_iris = iris.data, iris.target

#get database with only the first two attributes
X, y = X_iris[:,:2],y_iris

#split the database into a training and a testing set
#Test set will be the 25% taken randomly
X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=33)
print(X_train.shape,y_train.shape)

#Stardardize the features
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

colors = ['red','greenyellow','blue']
for i in range(len(colors)):
    xs = X_train[:, 0][y_train ==i]
    ys = X_train[:, 1][y_train ==i]
    plt.scatter(xs,ys,c=colors[i])
    
plt.legend(iris.target_names)
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.show()

clf = SGDClassifier()
clf.fit(X_train,y_train)

print(clf.coef_)
print()
print(clf.intercept_)

x_min,x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5

xs = np.arange(x_min,x_max,0.5)
fig, axes = plt.subplots(1,3)
fig.set_size_inches(10,6)
for i in [0,1,2]:
    axes[i].set_aspect("equal")
    axes[i].set_title('Class'+str(i)+'versus the rest')
    axes[i].set_xlabel('Sepal length')
    axes[i].set_ylabel('Sepal Width')
    axes[i].set_xlim(x_min,x_max)
    axes[i].set_ylim(y_min,y_max)
    plt.sca(axes[i])
    plt.scatter(X_train[:, 0],X_train[:, 1],c=y_train,cmap=plt.cm.prism)
    ys = (-clf.intercept_[i] - xs * clf.coef_[i, 0])/clf.coef_[i, 1]
    plt.plot(xs,ys,hold=True)
plt.show()

#PREDICCION
#florecilla que queremos predecir su tipo solo #dandole su altura y ancho: length y width
this = clf.predict(scaler.transform([[4.7,3.1]]))
print("Predicción: flor {}".format(iris.target_names[this[0]]))

print("bad idea testing on train data: ")
y_train_pred = clf.predict(X_train)
print("accuracy: ",metrics.accuracy_score(y_train,y_train_pred))
print()

print("good practice to test on test_data :)")
y_pred = clf.predict(X_test)
print("real accuracy: ", metrics.accuracy_score(y_test,y_pred))

#report of accuracy F1-Score, recall, prediction
print(metrics.classification_report(y_test,y_pred,target_names=iris.target_names))

#matrix of confusion
print(metrics.confusion_matrix(y_test,y_pred))
print()
"""To finish our evaluation process, we will introduce a very useful method known
as cross-validation. As we explained before, we have to partition our dataset into
a training set and a testing set. However, partitioning the data, results such that
there are fewer instances to train on, and also, depending on the particular partition
we make (usually made randomly), we can get either better or worse results.
Cross-validation allows us to avoid this particular case, reducing result variance and
producing a more realistic score for our models. The usual steps for k-fold
cross-validation are the following:
1. Partition the dataset into k different subsets.
2. Create k different models by training on k-1 subsets and testing on the
remaining subset.
3. Measure the performance on each of the k models and take the average
measure."""
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from scipy.stats import sem


#create a composite estimator made by a pipeline of the standarization and the linear model
"""With this
technique, we make sure that each iteration will standardize the data and then
train/test on the transformed data."""

clf = Pipeline([('scaler',preprocessing.StandardScaler()),('linear_model',SGDClassifier())])

#Create a k-fold cross validation iterator of k=5 folds 
cv = KFold(X.shape[0],5,shuffle=True,random_state=33)
#by default the score used is the one returned by score method of the stimator(accuracy)
scores = cross_val_score(clf,X,y,cv=cv)
print(scores)

def mean_score(scores):
    return ("Mean score: {0:.3f}(+/-{1:.3f})".format(np.mean(scores),sem(scores)))

print(mean_score(scores))  #The final average accuracy of our model