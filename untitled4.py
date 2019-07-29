# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 22:59:20 2018

@author: HP
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette('husl')

from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split

data=pd.read_csv('Iris.csv')

data.head()
data.info()
data.describe()

data['Species'].value_counts()

tmp=data.drop('Id',axis=1)
g=sns.pairplot(data, hue='Species', markers='+')
plt.show()

g=sns.violinplot(x='SepalLengthCm', y='Species', data=data, inner='quartile')
plt.show()
g=sns.violinplot(x='SepalWidthCm', y='Species', data=data, inner='quartile')
plt.show()
g=sns.violinplot(x='PetalLengthCm', y='Species', data=data, inner='quartile')
plt.show()
g=sns.violinplot(x='PetalWidthCm', y='Species', data=data, inner='quartile')
plt.show()

x=data.drop(['Id','Species'],axis=1)
y=data['Species']
print(x.shape)
print(y.shape)

k_range=list(range(1,26))
scores= []
for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(x,y)
    y_pred=knn.predict(x)
    scores.append(metrics.accuracy_score(y, y_pred))
    
plt.plot(k_range, scores)
plt.xlabel('value of k for KNN')
plt.ylabel('Accuracy Score')
plt.title('Accuracy Scores for values of k of k-Nearest-Neighbors')
plt.show()
logreg= LogisticRegression()
logreg.fit(x,y)
y_pred=logreg.predict(x)
print(metrics.accuracy_score(y,y_pred))

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.4,random_state=5)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

k_range=list(range(1,26))
scores= []
for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)
    y_pred=knn.predict(x_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))
plt.plot(k_range, scores)
plt.xlabel('value of k for KNN')
plt.ylabel('Accuracy Score')
plt.title('Accuracy Scores for values of k of k-Nearest-Neighbors')
plt.show()    
logreg= LogisticRegression()
logreg.fit(x_train,y_train)
y_pred=logreg.predict(x_test)
print(metrics.accuracy_score(y_test,y_pred))

knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(x,y)
knn.predict([[6,3,4,2]])

