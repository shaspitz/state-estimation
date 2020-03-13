'''
Created on Mar 13, 2020

@author: Shawn Marshall-Spitzbart

UC Berkeley ME231B Assignment #5 Problem 4
'''

import numpy as np
import matplotlib.pyplot as plt

# Implement time-varying linear system as given in problem statement
n = 2
A = np.array([[0.8, 0.6], [-0.6, 0.8]])
H = np.array([1, 0])

# Initial prior estimate and prior covariance of state
xp, Pp = 0, np.array([[3, 0], [0, 1]])

# Time-invariant process noise and measurement noise given as identity
V, W = np.eye(n, dtype = int), np.eye(n, dtype = int)

print(W)