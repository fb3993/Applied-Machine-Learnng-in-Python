#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 17:01:55 2020

@author: filber
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adspy_shared_utilities import plot_fruit_knn
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

pd.options.display.max_columns= None
pd.options.display.max_rows = None

cancer = load_breast_cancer()

# Question 1
# Convert the sklearn.dataset cancer to a DataFrame

def answer_one():
    
    return pd.DataFrame(np.c_[cancer['data'], cancer['target']],
                  columns= np.append(cancer['feature_names'], ['target']))

df = answer_one()

#Question 2
#What is the class distribution? (i.e. how many instances of malignant (encoded 0) and how many benign (encoded 1)?)
#This function should return a Series named target of length 2 with integer values and index = ['malignant', 'benign']

tumor_type = dict(zip(['malignant', 'benign'], [0, 1]))

def answer_two():
    # makes series
    instances_tumor = df['target'].value_counts()
    #reindex series
    instances_tumor = instances_tumor.rename({1: 'benign', 0: 'malignant'})
    
    return instances_tumor

#Question 3
#Split the DataFrame into X (the data) and y (the labels).
#This function should return a tuple of length 2: (X, y), where
#X, a pandas DataFrame, has shape (569, 30)
#y, a pandas Series, has shape (569,).

def answer_three():
    
#    df['label'] = np.where(df['target']==0, 'malignant', 'benign')
    X = df.drop(['target'], axis=1)
    y = df['target']
    
    return X, y

#Question 4
#Using train_test_split, split X and y into training and test sets (X_train, X_test, y_train, and y_test).
#Set the random number generator state to 0 using random_state=0 to make sure your results match the autograder!
#This function should return a tuple of length 4: (X_train, X_test, y_train, y_test), where
#X_train has shape (426, 30)
#X_test has shape (143, 30)
#y_train has shape (426,)
#y_test has shape (143,)
    
def answer_four():
    
    x, y = answer_three()
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 0)
    
    return x_train, x_test, y_train, y_test

#Question 5
#Using KNeighborsClassifier, fit a k-nearest neighbors (knn) classifier with X_train, y_train and using one nearest neighbor (n_neighbors = 1).
#This function should return a  sklearn.neighbors.classification.KNeighborsClassifier.
    
def answer_five():
    
    x_train, x_test, y_train, y_test = answer_four()
    knn = KNeighborsClassifier(n_neighbors = 1)
    return knn.fit(x_train, y_train)


#Question 6
#Using your knn classifier, predict the class label using the mean value for each feature.
#Hint: You can use cancerdf.mean()[:-1].values.reshape(1, -1) which gets the mean value for each feature, ignores the target column, and reshapes the data from 1 dimension to 2 (necessary for the precict method of KNeighborsClassifier).
#This function should return a numpy array either array([ 0.]) or array([ 1.])
    
def answer_six():
    
    knn = answer_five()
    means = df.mean()[:-1].values.reshape(1, -1)
    cancer_prediction = knn.predict(means)
    
    return cancer_prediction

#Question 7
#Using your knn classifier, predict the class labels for the test set X_test.
#This function should return a numpy array with shape (143,) and values either 0.0 or 1.0.
    
def answer_seven():
    
    x_train, x_test, y_train, y_test = answer_four()
    knn = answer_five()
    cancer_predictions= knn.predict(t_test)
    return cancer_predictions

#Question 8
#Find the score (mean accuracy) of your knn classifier using X_test and y_test.
#This function should return a float between 0 and 1
    
def answer_eight():
    
    x_train, x_test, y_train, y_test = answer_four()
    knn = answer_five()
    
    return knn.score(x_test, y_test)

#Optional plot
#Try using the plotting function below to visualize the differet predicition scores between training and test sets, as well as malignant and benign cells.
    
def accuracy_plot():
    
    x_train, x_test, y_train, y_test = answer_four()
    knn = answer_five()
    
    # find the training and testing accuracies by target value (i.e. malignant, benign)
    mal_train_x = x_train[y_train==0]
    mal_train_y = y_train[y_train==0]
    ben_train_x = x_train[y_train==1]
    ben_train_y = y_train[y_train==1]    
    
    mal_test_x = x_test[y_test==0]
    mal_test_y = y_test[y_test==0]
    ben_test_x = x_test[y_test==1]
    ben_test_y = y_test[y_test==1]

    scores = [knn.score(mal_train_x, mal_train_y), knn.score(ben_train_x, ben_train_y),
              knn.score(mal_test_x, mal_test_y), knn.score(ben_test_x, ben_test_y)]

    plt.figure()

    # Plot the scores as a bar chart
    bars = plt.bar(np.arange(4), scores, color=['#4c72b0','#4c72b0','#55a868','#55a868'])

    # directly label the score onto the bars
    for bar in bars:
        height = bar.get_height()
        plt.gca().text(bar.get_x() + bar.get_width()/2, height*.90, '{0:.{1}f}'.format(height, 2), 
                     ha='center', color='w', fontsize=11)

    # remove all the ticks (both axes), and tick labels on the Y axis
    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')

    # remove the frame of the chart
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    plt.xticks([0,1,2,3], ['Malignant\nTraining', 'Benign\nTraining', 'Malignant\nTest', 'Benign\nTest'], alpha=0.8);
    plt.title('Training and Test Accuracies for Malignant and Benign Cells', alpha=0.8)
                   
                
    
    
    
    
    
    
    
    
    
    
    