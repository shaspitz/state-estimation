'''
Created on Mar 19, 2020

@author: Shawn Marshall-Spitzbart

UC Berkeley ME231B Assignment #5 Problem 2
'''

import numpy as np

''' This problem involves the implementation of a time-varying Kalman filter
 for a simple (scalar) system with uniform process and measurement noises '''

# First define global vars

# Time-invariant linear system as given in problem statement
N = 1
A = 1
H = 1

# x(0), v(k), and w(k) are all uniformly distributed in the range [-1, 1]
a, b = -1, 1

# Time-invariant process and measurement noise covariances (uniform)
V, W = (b - a)**2/12, (b - a) ** 2/12

# Process and measurement noise are zero mean, white, and independant

''' Function that returns value corresponding to uniform dist matching
 required mean/variance for process and measurement noises '''


def r_uniform(): return np.random.uniform(a, b)

# Define measurement and time update for Kalman filter implementation


def time_update(xm, Pm):
    xp = A*xm
    Pp = A*Pm*A + V
    return xp, Pp


def meas_update(xp, Pp, z):
    K = Pp*H*(H*Pp*H + W)**-1
    xm = xp + K*(z - H*xp)
    Pm = (np.eye(N) - K*H)*Pp*(np.eye(N) - K*H) + K*W*K
    return xm, Pm


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
    + repr(np.round(xm, 4))
    )

print(
    'With Pm at time k = 1 being: '
    + repr(np.round(Pm[0, 0], 4))
    )

print('The KF estimate here of 2/3 coincides with the MMSE estimate from'
      ' problem 1. This makes sense intuitively because amongst the class'
      ' of linear and unbiased estimators, the Kalman filter optimally'
      ' minimizes the mean squared estimation error. The difference in'
      ' implementation between a standard KF and the optimal estimator from'
      ' problem 1 is that we do not compute the entire state pdf at each'
      ' timestep for a KF, only its MMSE estimate and variance.')
