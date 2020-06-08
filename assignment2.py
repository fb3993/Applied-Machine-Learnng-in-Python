#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 11:14:13 2020

@author: filber
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from adspy_shared_utilities import plot_decision_tree

from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

"""
Part 1 - REGRESSION
"""

np.random.seed(0)
n = 15
x = np.linspace(0,10,n) + np.random.randn(n)/5
y = np.sin(x)+x/6 + np.random.randn(n)/10

X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)

def part1_scatter():
    plt.figure()
    plt.scatter(X_train, y_train, label='training data')
    plt.scatter(X_test, y_test, label='test data')
    plt.legend(loc=4)
    
    
# =============================================================================
# Question 1
# Write a function that fits a polynomial LinearRegression model on the training
#  data X_train for degrees 1, 3, 6, and 9. 
#  (Use PolynomialFeatures in sklearn.preprocessing to create the polynomial 
#  features and then fit a linear regression model) For each model, find 100 
#  predicted values over the interval x = 0 to 10 (e.g. np.linspace(0,10,100)) 
#  and store this in a numpy array. The first row of this array should correspond
#  to the output from the model trained on degree 1, the second row degree 3, 
#  the third row degree 6, and the fourth row degree 9.
# =============================================================================
def answer_one():
    
    result = np.zeros([len([1, 3, 6, 9]), 100])
    predict = np.linspace(0, 10, 100).reshape(-1, 1)
    X_tr = X_train.reshape(-1, 1)
    
    for i, deg in enumerate([1, 3, 6, 9]):
        poly = PolynomialFeatures(degree=deg)
        X_ = poly.fit_transform(X_tr)
        predict_ = poly.fit_transform(predict)
        reg = LinearRegression()
        reg.fit(X_, y_train)
        result[i, :] = reg.predict(predict_)
    
    return result

answer_one()

def plot_one(degree_predictions):
    plt.figure(figsize=(10,5))
    plt.plot(X_train, y_train, 'o', label='training data', markersize=10)
    plt.plot(X_test, y_test, 'o', label='test data', markersize=10)
    for i,degree in enumerate([1,3,6,9]):
        plt.plot(np.linspace(0,10,100), degree_predictions[i], alpha=0.8, lw=2, label='degree={}'.format(degree))
    plt.ylim(-1,2.5)
    plt.legend(loc=4)
    
# =============================================================================
# Question 2
# Write a function that fits a polynomial LinearRegression model on the training
#  data X_train for degrees 0 through 9. For each model compute the  R2
#  (coefficient of determination) regression score on the training data as well
#  as the the test data, and return both of these arrays in a tuple.
# This function should return one tuple of numpy arrays (r2_train, r2_test). 
# Both arrays should have shape (10,)
# =============================================================================

def answer_two():
    degrees = [0,1,2,3,4,5,6,7,8,9]
    r2_train = np.zeros([len(degrees)])
    r2_test = np.zeros([len(degrees)])
    
    X_tr = X_train.reshape(-1, 1)
    X_ts = X_test.reshape(-1, 1)
    
    for i, deg in enumerate(degrees):
        pol = PolynomialFeatures(degree=deg)
        X_tra = pol.fit_transform(X_tr)
        X_tes = pol.transform(X_ts)
        reg = LinearRegression()
        reg.fit(X_tra, y_train)
        r2_train[i] = r2_score(y_train, reg.predict(X_tra))
        r2_test[i] = r2_score(y_test, reg.predict(X_tes))
        
    return r2_train, r2_test

# =============================================================================
# Question 3
# Based on the  R2 scores from question 2 (degree levels 0 through 9), 
# what degree level corresponds to a model that is underfitting?
# What degree level corresponds to a model that is overfitting? 
# What choice of degree level would provide a model with good 
# generalization performance on this dataset?
# Hint: Try plotting the R2 scores from question 2 to visualize the 
# relationship between degree level and R2. 
# Remember to comment out the import matplotlib line before submission.
# This function should return one tuple with the degree values in this order: 
# (Underfitting, Overfitting, Good_Generalization). 
# There might be multiple correct solutions, however, 
# you only need to return one possible solution, for example, (1,2,3).
# =============================================================================

def answer_three():
    r2_train, r2_test = answer_two()
    r2_train_sort = np.sort(r2_train)
    r2_test_sort = np.sort(r2_test)
    
    underfitting = 0
    overfitting = 0
    good_generalization = 0
    
    for deg, data in enumerate(zip(r2_train, r2_test)):
        if data[0] < r2_train_sort[5] and data[1] < r2_test_sort[5]:
            underfitting = deg
        if data[0] >= r2_train_sort[5] and data[1] < r2_test_sort[5]:
            overfitting = deg
        if data[0] >= r2_train_sort[5] and data[1] >= r2_test_sort[7]:
            good_generalization = deg
    
    return underfitting, overfitting, good_generalization


# =============================================================================
# Question 4
# Training models on high degree polynomial features can result in overly 
# complex models that overfit, so we often use regularized versions of the model 
# to constrain model complexity, as we saw with Ridge and Lasso linear regression.
# For this question, train two models: a non-regularized LinearRegression model 
# (default parameters) and a regularized Lasso Regression model 
# (with parameters alpha=0.01, max_iter=10000) both on polynomial features 
# of degree 12. Return the $R^2$ score for both the LinearRegression and 
# Lasso model's test sets.
# This function should return one tuple 
# (LinearRegression_R2_test_score, Lasso_R2_test_score)
# =============================================================================

def answer_four():
    X_tr = X_train.reshape(-1,1)
    X_ts = X_test.reshape(-1,1)
    pol = PolynomialFeatures(degree = 12)
    X_tra = pol.fit_transform(X_tr)
    X_tes = pol.transform(X_ts)
    
    reg = LinearRegression()
    reg.fit(X_tra, y_train)
    R2_reg_score = r2_score(y_test, reg.predict(X_tes))
    
    lasso = Lasso(alpha=0.01, max_iter=10000).fit(X_tra, y_train)
    R2_lasso_score = r2_score(y_test, lasso.predict(X_tes))
    
    return R2_reg_score, R2_lasso_score


"""
Part 2 - CLASSIFICATION
"""

mush_df = pd.read_csv('mushrooms.csv')
mush_df2 = pd.get_dummies(mush_df)
X_mush = mush_df2.iloc[:,2:]
y_mush = mush_df2.iloc[:,1]

# use the variables X_train2, y_train2 for Question 5
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_mush, y_mush, random_state=0)

#For performance reasons in Question 6 and 7, we will create a smaller version of the
#entire mushroom dataset for use in those questions. For simplicity we'll just re-use
#the 25% test split create above as the representative subset.

# Use the variables X_subset, y_subset for questions 6 and 7.
X_subset = X_test2
y_subset = y_test2


# =============================================================================
# Question 5
# Using X_train2 and y_train2 from the preceeding cell, train a 
# DecisionTreeClassifier with default parameters and random_state=0. 
# What are the 5 most important features found by the decision tree?
# As a reminder, the feature names are available in the X_train2.columns property,
# and the order of the features in X_train2.columns matches the order of the
# feature importance values in the classifier's feature_importances_ property.
# This function should return a list of length 5 containing
# the feature names in descending order of importance.
# Note: remember that you also need to set random_state 
# in the DecisionTreeClassifier.
# =============================================================================

def answer_five():
    clf = DecisionTreeClassifier().fit(X_train2, y_train2)
    features = []
    for index, importance in enumerate(clf.feature_importances_):
        features.append([importance, X_train2.columns[index]])
    features.sort(reverse=True)
    features = np.array(features)
    features = features[:5,1]
    features = features.tolist()
    
    return features
    
#Question 6
#For this question, we're going to use the validation_curve function in sklearn.model_selection to determine training and test scores for a Support Vector Classifier (SVC) with varying parameter values. Recall that the validation_curve function, in addition to taking an initialized unfitted classifier object, takes a dataset as input and does its own internal train-test splits to compute results.
#Because creating a validation curve requires fitting multiple models, for performance reasons this question will use just a subset of the original mushroom dataset: please use the variables X_subset and y_subset as input to the validation curve function (instead of X_mush and y_mush) to reduce computation time.
#The initialized unfitted classifier object we'll be using is a Support Vector Classifier with radial basis kernel. So your first step is to create an SVC object with default parameters (i.e. kernel='rbf', C=1) and random_state=0. Recall that the kernel width of the RBF kernel is controlled using the gamma parameter.
#With this classifier, and the dataset in X_subset, y_subset, explore the effect of gamma on classifier accuracy by using the validation_curve function to find the training and test scores for 6 values of gamma from 0.0001 to 10 (i.e. np.logspace(-4,1,6)). Recall that you can specify what scoring metric you want validation_curve to use by setting the "scoring" parameter. In this case, we want to use "accuracy" as the scoring metric.
#For each level of gamma, validation_curve will fit 3 models on different subsets of the data, returning two 6x3 (6 levels of gamma x 3 fits per level) arrays of the scores for the training and test sets.
#Find the mean score across the three models for each level of gamma for both arrays, creating two arrays of length 6, and return a tuple with the two arrays.
#e.g.
#if one of your array of scores is
#array([[ 0.5,  0.4,  0.6],
#       [ 0.7,  0.8,  0.7],
#       [ 0.9,  0.8,  0.8],
#       [ 0.8,  0.7,  0.8],
#       [ 0.7,  0.6,  0.6],
#       [ 0.4,  0.6,  0.5]])
#it should then become
#array([ 0.5,  0.73333333,  0.83333333,  0.76666667,  0.63333333, 0.5])
#This function should return one tuple of numpy arrays (training_scores, test_scores) where each array in the tuple has shape (6,).

def answer_six():
    
    svc = SVC(kernel='rbf', C=1, random_state=1)
    gamma = np.logspace(-4, 1, 6)
    train_scores, test_scores = validation_curve(svc, X_subset, y_subset, param_name='gamma', param_range=gamma, scoring='accuracy')
    scores = (train_scores.mean(axis=1), test_scores.mean(axis=1))
    
    return scores


#Question 7
#Based on the scores from question 6, what gamma value corresponds to a model that is underfitting (and has the worst test set accuracy)? What gamma value corresponds to a model that is overfitting (and has the worst test set accuracy)? What choice of gamma would be the best choice for a model with good generalization performance on this dataset (high accuracy on both training and test set)?
#Hint: Try plotting the scores from question 6 to visualize the relationship between gamma and accuracy. Remember to comment out the import matplotlib line before submission.
#This function should return one tuple with the degree values in this order: (Underfitting, Overfitting, Good_Generalization) Please note there is only one correct solution.

def answer_seven():
    return (0.001, 10, 0.1)



























