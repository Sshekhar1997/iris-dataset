# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 17:00:22 2018

@author: shashank
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.cluster import KMeans

style.use('ggplot')

x=np.array([[1,2],
           [1.5,1.8],
           [5,8],
           [8,8],
           [1,0.6],
           [9,11]])
           
           
           
          

plt.scatter(x[:, 0],x[:, 1],s=150, linewidth=5, zorder=10)
plt.show()


clf=KMeans(n_clusters=2)
clf.fit(x)
centroid=clf.cluster_centers_
labels=clf.labels_

colors=['g.','r.','c.','y.']
for i in range(len(x)):
    plt.plot(x[i][0],x[i][1],colors[labels[i]],marksize=
             10)
    plt.scatter(centroid[:, 0], centroid[:, 1],marker='+',s=150, linewidth=5, zorder=10)
    plt.show()
