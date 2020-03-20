'''
Created on Mar 19, 2020

@author: shawn

UC Berkeley ME231B Assignment #5 Problem 2
'''

import numpy as np

''' This problem involves the implementation of a time-varying Kalman filter
 for a simple (scalar) system with uniform process and measurement noises '''

# Time-invariant linear system as given in problem statement
global N, A, H
N = 1
A = 1
H = 1

# x(0), v(k), and w(k) are all uniformly distributed in the range [-1, 1]
global a, b
a, b = -1, 1

# Time-invariant process and measurement noise covariances (uniform)
global V, W
V, W = (b - a)**2/12, (b - a) ** 2/12

# Mean of process noise and measurement noise are zero from range [-1, 1]
E_v, E_w = 0, 0

''' Function that returns value corresponding to uniform dist matching
 required mean/variance for process and measurement noises '''


def r_uniform(): return np.random.uniform(a, b)

# Define measurement and time update for Kalman filter implementation


def meas_update(xp, Pp, z):
    K = Pp*H*(H*Pp*H + W)**-1
    xm = xp + K*(z - H*xp)
    Pm = (np.eye(N) - K*H)*Pp*(np.eye(N) - K*H) + K*W*K
    return xm, Pm


def time_update(xm, Pm):
    xp = A*xm
    Pp = A*Pm*A + V
    return xp, Pp


# Initialize estimate and covariance of state (at k = 0)
xm, Pm = 0, (b - a) ** 2/12

# Find KF estimate at time k = 1 given that z(1) = 1
z = 1

# Kalman filter estimation: time update
xp, Pp = time_update(xm, Pm)

# Kalman filter estimation: measurement update
xm, Pm = meas_update(xp, Pp, z)

print(
    'Kalman filter estimate at time k = 1 is: '
    + repr(round(xm, 4))
    )

print('Comments:')