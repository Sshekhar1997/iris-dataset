#Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import *
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap

#Importing the Dataset
iris=load_iris()
X=iris.data
y=iris.target

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

## Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

'''                         K NEAREST NEIGHBOURS CLASSIFICATION IMPLEMENTATION
                        ----------------------------------------------------------
'''

# Fitting K-Nearest Neighbours to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier_knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier_knn.fit(X_train, y_train)

# Predicting the Test set results using K-NN Algorithm
y_pred_knn = classifier_knn.predict(X_test)

# Making the Confusion Matrix for K-NN
cm_knn = confusion_matrix(y_test, y_pred_knn)

#Preparing another classifier for the Visualisation that trains the Classifier on Sepal characteristics
classifier_knnvis= KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier_knnvis.fit(X_train[:,[0,1]], y_train)

# Visualising the Test set results of K-NN
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier_knnvis.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
            alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green','blue'))(i), label = j)
plt.title('K-Nearest Neighbours Classification (Test set)')
plt.xlabel('Sepal Length(in cm)')
plt.ylabel('Sepal Width(in cm)')
L=plt.legend()
L.get_texts()[0].set_text('Setosa')
L.get_texts()[1].set_text('Versicolor')
L.get_texts()[2].set_text('Virginica')
plt.show()

'''                                 NAIVE BAYE'S CLASSIFICATION
                                 --------------------------------
'''

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier_nb = GaussianNB()
classifier_nb.fit(X_train, y_train)

# Predicting the Test set results using Naive Bayes Algorithm
y_pred_nb = classifier_nb.predict(X_test)

# Making the Confusion Matrix for Naive Bayes
cm_nb = confusion_matrix(y_test, y_pred_nb)

#Preparing another classifier for the Visualisation that trains the Classifier on Sepal characteristics
classifier_nbvis= classifier_nb = GaussianNB()
classifier_nbvis.fit(X_train[:,[0,1]], y_train)

# Visualising the Test set results of K-NN
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier_nbvis.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
            alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green','blue'))(i), label = j)
plt.title('Naive Bayes Classification (Test set)')
plt.xlabel('Sepal Length(in cm)')
plt.ylabel('Sepal Width(in cm)')
L=plt.legend()
L.get_texts()[0].set_text('Setosa')
L.get_texts()[1].set_text('Versicolor')
L.get_texts()[2].set_text('Virginica')
plt.show()

'''                                   SUPPORT VECTOR MACHINE CLASSIFICATION
                                    -----------------------------------------
'''

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier_svm = SVC(kernel = 'linear', random_state = 0)
classifier_svm.fit(X_train, y_train)

# Predicting the Test set results using SVM
y_pred_svm = classifier_svm.predict(X_test)

# Making the Confusion Matrix for SVM
cm_svm = confusion_matrix(y_test, y_pred_svm)

#Preparing another classifier for the Visualisation that trains the Classifier on Sepal characteristics
classifier_svmvis= classifier_nb = SVC(kernel = 'linear', random_state = 0)
classifier_svmvis.fit(X_train[:,[0,1]], y_train)

# Visualising the Test set results of K-NN
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier_svmvis.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
            alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green','blue'))(i), label = j)
plt.title('Support Vector Machine Classification (Test set)')
plt.xlabel('Sepal Length(in cm)')
plt.ylabel('Sepal Width(in cm)')
L=plt.legend()
L.get_texts()[0].set_text('Setosa')
L.get_texts()[1].set_text('Versicolor')
L.get_texts()[2].set_text('Virginica')
plt.show()

'''                               KERNEL SUPPORT VECTOR MACHINE CLASSIFICATION
                                ------------------------------------------------
'''

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier_ksvm = SVC(kernel = 'rbf', random_state = 0)
classifier_ksvm.fit(X_train, y_train)

# Predicting the Test set results using Kernel SVM
y_pred_ksvm = classifier_ksvm.predict(X_test)

# Making the Confusion Matrix for Kernel SVM
cm_ksvm = confusion_matrix(y_test, y_pred_ksvm)

#Preparing another classifier for the Visualisation that trains the Classifier on Sepal characteristics
classifier_ksvmvis= SVC(kernel = 'rbf', random_state = 0)
classifier_ksvmvis.fit(X_train[:,[0,1]], y_train)

# Visualising the Test set results of K-NN
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier_ksvmvis.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
            alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green','blue'))(i), label = j)
plt.title('Kernel Support Vector Machine Classification (Test set)')
plt.xlabel('Sepal Length(in cm)')
plt.ylabel('Sepal Width(in cm)')
L=plt.legend()
L.get_texts()[0].set_text('Setosa')
L.get_texts()[1].set_text('Versicolor')
L.get_texts()[2].set_text('Virginica')
plt.show()

'''                              RANDOM FOREST CLASSIFICATION
                               --------------------------------
'''

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier_rf.fit(X_train, y_train)

# Predicting the Test set results using Random Forest Classification
y_pred_rf = classifier_rf.predict(X_test)

# Making the Confusion Matrix for Random Forest Classification
cm_rf = confusion_matrix(y_test, y_pred_rf)

#Preparing another classifier for the Visualisation that trains the Classifier on Sepal characteristics
classifier_rfvis= RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier_rfvis.fit(X_train[:,[0,1]], y_train)

# Visualising the Test set results of K-NN
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier_rfvis.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
            alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green','blue'))(i), label = j)
plt.title('Random Forest Classification (Test set)')
plt.xlabel('Sepal Length(in cm)')
plt.ylabel('Sepal Width(in cm)')
L=plt.legend()
L.get_texts()[0].set_text('Setosa')
L.get_texts()[1].set_text('Versicolor')
L.get_texts()[2].set_text('Virginica')
plt.show()

'''                                       ACCURACY OF THE CLASSIFIERS
                                        -------------------------------
'''
from sklearn.metrics import accuracy_score
knn_acc = accuracy_score(y_test,y_pred_knn)
nb_acc = accuracy_score(y_test,y_pred_nb)
svm_acc = accuracy_score(y_test,y_pred_svm)
ksvm_acc = accuracy_score(y_test,y_pred_ksvm)
rf_acc = accuracy_score(y_test,y_pred_rf)
barlist=plt.bar([1,2,3,4,5],height=[knn_acc, nb_acc, svm_acc, ksvm_acc,rf_acc])
plt.xticks([1.45,2.45,3.45,4.45,5.45], ['K-Nearest\nNeighbours','Naive\nBayes','Support\nVector\nMachine',
           'Kernel\nSupport\nVector\nMachine','Random\nForest'])
barlist[0].set_color('r')
barlist[1].set_color('b')
barlist[2].set_color('g')
barlist[3].set_color('y')
barlist[4].set_color('k')
plt.xlabel('Type Of Classification')
plt.ylabel('Accuracy of Classifier')
plt.title('ACCURACY OF THE IMPLEMENTED CLASSIFIERS')
plt.show()
