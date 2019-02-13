#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 22:08:27 2019

@author: Adrian
"""

from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

#list of four possible inputs to gate
data = [[0,0], [0,1], [1,0], [1,1]]
#AND, OR, and XOR gate labels
ANDlabels = [0, 0, 0, 1]
ORlabels = [0, 1, 1, 1]
XORlabels = [0, 1, 1, 0]

#generate x and y values from data
x_values = [point[0] for point in data]
y_values = [point[1] for point in data]

#plot AND gate
fig = plt.figure(figsize = (12,3))
plt.subplot(1,3,1)
plt.scatter(x_values, y_values, c=ANDlabels)
plt.title("AND Logic Gate")

#plot OR gate
plt.subplot(1,3,2)
plt.scatter(x_values, y_values, c=ORlabels)
plt.title("OR Logic Gate")

#plot XOR gate
plt.subplot(1,3,3)
plt.scatter(x_values, y_values, c=XORlabels)
plt.title("XOR Logic Gate")

plt.show()

##########################
#Conclusion from above three graphs:
#AND and OR gates are linearly separable because you can draw a straight line that completely separates the points of each class.
#XOR gate is NOT linearly seperable because you can't draw a straight line that completely separates the points of each class.
###########################

#create perceptron object, train model, and print accuracy of model on the data points 
classifier = Perceptron(max_iter=40, tol=1e-3)

classifier.fit(data, ANDlabels)
print(classifier.score(data, ANDlabels))
#output of 1.0 indicates that 100% of the time, model was able to correctly determine output given data

classifier.fit(data, ORlabels)
print(classifier.score(data, ORlabels))
#output of 1.0 indicates that 100% of the time, model was able to correctly determine output given data

classifier.fit(data, XORlabels)
print(classifier.score(data, XORlabels))
#output of 0.5 indicates that 50% of the time, model was able to correctly determine output given data


##########################

#decision fucntion can tell us the proximity of a particular point from the decision boundary
#example:
print(classifier.decision_function([[0, 0], [1, 1], [0.5, 0.5]]))
#output of [ 0.  -1.  -0.5] tells us that points [0, 0], [1, 1], [0.5, 0.5] are 0, -1.0, and-0.5 units away from XOR decision boundary

##########################


#use decision_function method to create a heatmap containing 100 equidistant, ordered pairs and their respective distances from the decision boundary
x_values = np.linspace(0, 1, 100)
y_values = np.linspace(0, 1, 100)
point_grid = list(product(x_values, y_values))

#plot heatmap for AND gate
classifier.fit(data, ANDlabels)
distances = classifier.decision_function(point_grid)
abs_distances = [abs(pt) for pt in distances]
distances_matrix = np.reshape(abs_distances, (100,100))

heatmap = plt.pcolormesh(x_values, y_values, distances_matrix)
cbar = plt.colorbar(heatmap)
plt.xlabel("X-Value")
plt.ylabel("Y-Value")
plt.title("AND Logic Gate Heatmap")
cbar.set_label("Distance From Decision Boundary", rotation=270, labelpad=13)
plt.show()

#plot heatmap for OR gate
classifier.fit(data, ORlabels)
distances = classifier.decision_function(point_grid)
abs_distances = [abs(pt) for pt in distances]
distances_matrix = np.reshape(abs_distances, (100,100))

heatmap = plt.pcolormesh(x_values, y_values, distances_matrix)
cbar = plt.colorbar(heatmap)
plt.xlabel("X-Value")
plt.ylabel("Y-Value")
plt.title("OR Logic Gate Heatmap")
cbar.set_label("Distance From Decision Boundary", rotation=270,labelpad=13)
plt.show()

#plot heatmap for XOR gate
classifier.fit(data, XORlabels)
distances = classifier.decision_function(point_grid)
abs_distances = [abs(pt) for pt in distances]
distances_matrix = np.reshape(abs_distances, (100,100))

heatmap = plt.pcolormesh(x_values, y_values, distances_matrix)
cbar = plt.colorbar(heatmap)
plt.xlabel("X-Value")
plt.ylabel("Y-Value")
plt.title("XOR Logic Gate Heatmap")
cbar.set_label("Distance From Decision Boundary", rotation=270, labelpad=13)
plt.show()

