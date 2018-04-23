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
X,y = X_iris[:,:2],y_iris

#split the database into a training and a testing set
#Test set will be the 25% taken randomly
"""Why not just select the first 112 examples?
This is because it could happen that the instance ordering within the sample could
matter and that the first instances could be different to the last ones. In fact, if you
look at the Iris datasets, the instances are ordered by their target class, and this
implies that the proportion of 0 and 1 classes will be higher in the new training set,
compared with that of the original dataset. We always want our training data to be a
representative sample of the population they represent."""
X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=33)
print(X_train.shape,y_train.shape)

#Stardardize the features
"""The last three lines modify the training set in a process usually
called feature scaling. For each feature, calculate the average, subtract the mean
value from the feature value, and divide the result by their standard deviation"""
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
"""After
scaling, each feature will have a zero average, with a standard deviation of one. This
standardization of values (which does not change their distribution, as you could
verify by plotting the X values before and after scaling) is a common requirement of
machine learning methods, to avoid that features with large values may weight too
much on the final results."""

colors = ['red','greenyellow','blue']
for i in range(len(colors)):
    xs = X_train[:, 0][y_train ==i]
    ys = X_train[:, 1][y_train ==i]
    plt.scatter(xs,ys,c=colors[i])
    
plt.legend(iris.target_names)
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.show()

"""To implement linear classification, we will use the SGDClassifier from scikit-learn.
SGD stands for Stochastic Gradient Descent, a very popular numerical procedure
to find the local minimum of a function (in this case, the loss function, which
measures how far every instance is from our boundary). The algorithm will learn the
coefficients of the hyperplane by minimizing the loss function."""

clf = SGDClassifier()
clf.fit(X_train,y_train)

"""The fit
function is probably the most important one in scikit-learn. It receives the training
data and the training classes, and builds the classifier. Every supervised learning
method in scikit-learn implements this function."""
print(clf.coef_)
print()
print(clf.intercept_)
"""Indeed in the real plane, with these three values, we can draw a line, represented by
the following equation:
-17.62477802 - 28.53692691 * x1 + 15.05517618 * x2 = 0
Now, given x1 and x2 (our real-valued features), we just have to compute the value
of the left-side of the equation: if its value is greater than zero, then the point is
above the decision boundary (the red side), otherwise it will be beneath the line (the
green or blue side). Our prediction algorithm will simply check this and predict the
corresponding class for any new iris flower."""

"""The following code draws the three decision boundaries and lets us know if they
worked as expected:"""
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

"""This figure tells us that 82 percent of the training set instances are correctly classified
by our classifier.
Probably, the most important thing you should learn from this chapter is that
measuring accuracy on the training set is really a bad idea. You have built your
model using this data, and it is possible that your model adjusts well to them but
performs poorly in future (previously unseen data), which is its purpose. This
phenomenon is called overfitting"""

print("bad idea testing on train data: ")
y_train_pred = clf.predict(X_train)
print("accuracy: ",metrics.accuracy_score(y_train,y_train_pred))
print()

"""So, never measure based on your training data.
This is why we have reserved part of the original dataset (the testing partition)—we
want to evaluate performance on previously unseen data. Let's check the accuracy
again, now on the evaluation set (recall that it was already scaled):"""
print("good practice to test on test_data :)")
y_pred = clf.predict(X_test)
print("real accuracy: ", metrics.accuracy_score(y_test,y_pred))

"""
TP= True Positives
TN=true negatives
FP=False Positives
FN=False Negatives
With m being the sample size (that is, TP + TN + FP + FN), we have the
following formulae:
• Accuracy = (TP + TN) / m
• Precision = TP / (TP + FP)
• Recall = TP / (TP + FN)
• F1-score = 2 * Precision * Recall / (Precision + Recall)"""
"""support is how many instances of each class we had in the teting set"""
print(metrics.classification_report(y_test,y_pred,target_names=iris.target_names))

"""Another useful metric (especially for multi-class problems) is the confusion matrix:
in its (i, j) cell, it shows the number of class instances i that were predicted to
be in class j. A good classifier will accumulate the values on the confusion matrix
diagonal, where correctly classified instances belong."""

print(metrics.confusion_matrix(y_test,y_pred))

"""Our classifier is never wrong in our evaluation set when it classifies class 0 (setosa)
flowers. But, when it faces classes 1 and 2 flowers (versicolor and virginica), it
confuses them. The confusion matrix gives us useful information to know what types
of errors the classifier is making."""


