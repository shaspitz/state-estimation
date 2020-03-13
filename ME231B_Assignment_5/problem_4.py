'''
Created on Mar 13, 2020

@author: Shawn Marshall-Spitzbart

UC Berkeley ME231B Assignment #5 Problem 4
'''

import numpy as np
import matplotlib.pyplot as plt

# This problem involves the implementation of a time-varying Kalman filter for a system with uniform (ie non-Gaussian) process and measurement noises

# Time-varying linear system as given in problem statement
N = 2
A = np.array([[0.8, 0.6], [-0.6, 0.8]])
H = np.array([1, 0])

# Initialize prior estimate and prior covariance of state (at k = 0), and compute first time update 
xm, Pm = 0, np.array([[3, 0], [0, 1]])

# Time-invariant process and measurement noises covariance given as identity matrix
global V, W
V, W = np.eye(N, dtype = int), np.eye(N, dtype = int)

# Mean of process noise and measurement noise given as zero
E_v, E_w = 0, 0

T_f = 11 # Simulation Timesteps (0 included)
sim_tot = 10000 # Number of simulations

# preallocate parameters
e = np.zeros(sim_tot, T_f)

def time_update(A, xm, Pm):
    x = A*xm
    Pp = A*Pm*np.transpose(A) + np.random.uniform()

for sim in range(0, sim_tot):
    for k in range(0, T_f):
        # Simulate system
        
        # Kalman filter estimation: measurement update
        
        # Kalman filter estimation: time update