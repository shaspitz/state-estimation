'''
Created on Mar 18, 2020

@author: Shawn Marshall-Spitzbart

UC Berkeley ME231B Assignment #5 Problem 2
'''

import numpy as np
import matplotlib.pyplot as plt

''' This problem involves the implementation of a time-varying Kalman filter
 for a system with Gaussian process and measurement noises '''

# Time-invariant linear system as given in problem statement
global N, A, H
N = 2
A = np.matrix([[0.8, 0.6], [-0.6, 0.8]])
H = np.matrix([1, 0])

# Time-invariant process and measurement noises covariance given as identity matrix
global V, W
V, W = np.eye(N), np.eye(1)

# Mean of process noise and measurement noise given as zero
E_v, E_w = 0, 0

'''Function that returns value corresponding to Gaussian dist matching
 required mean/variance for process and measurement noises.
 Note that mean = 0 and var = I for both v(k) and w(k)'''
r_normal = lambda: np.random.normal(0, 1, 1)

# Define measurement and time update for Kalman Filter implementation


def meas_update(xp, Pp, z):
    K = Pp*H.transpose()*np.linalg.inv(H*Pp*H.transpose() + W)
    xm = xp + K*(z - H*xp)
    Pm = (np.eye(N) - K*H)*Pp*np.transpose(np.eye(N) - K*H) + K*W*K.transpose()
    return xm, Pm


def time_update(xm, Pm):
    xp = A*xm
    Pp = A*Pm*A.transpose() + V
    return xp, Pp


# Initialize estimate and covariance of state (at k = 0)
xm_0, Pm_0 = np.zeros([2, 1]), np.matrix([[3, 0], [0, 1]])

# (a) Compute the PDF of y(1) given the observation z(1) = 2.22
# First simulate prediction and measurement update steps of KF forward to k = 1
xp, Pp = time_update(xm_0, Pm_0)
xm, Pm = meas_update(xp, Pp, 2.22)

'''
Since output y is an affine linear transformation of GRV x(k), y(k) itself
is a GRV with mean [1 1]*xm(k) and variance [1 1]*Pm(k)*[1 1]' (see writing)
'''
E_y = lambda xm: np.matrix([[1, 1]])*xm
Var_y = lambda Pm: np.matrix([[1, 1]])*Pm*np.matrix([[1, 1]]).transpose()

E_y_val = E_y(xm)
Var_y_val = Var_y(Pm)

print(
    'Answer to (a): y(1) is normally distributed with mean: '
    + repr(round(E_y_val[0, 0], 4))
    + ' and variance: '
    + repr(round(Var_y_val[0, 0], 4))
    )

T_f = 11  # Simulation Timesteps (0 included)

# Initialize variance storage array with first two components at k = 0, 1
Var_y_vec = np.zeros([T_f, 1])
Var_y_vec[0] = Var_y(Pm_0)
Var_y_vec[1] = Var_y_val

for k in range(2, T_f):  # Note that estimates for k = 0, 1 is initilized above

    # Simulate system and measurement
    v_k = np.matrix([r_normal(), r_normal()]).transpose()  # Process noise
    w_k = np.matrix([r_normal()])  # Scalar measurement noise
    x_true = A*xm + v_k
    z = H*x_true + w_k

    # Kalman filter estimation: time update
    xp, Pp = time_update(xm, Pm)

    # Kalman filter estimation: measurement update
    xm, Pm = meas_update(xp, Pp, z)

    # Compute variance of y(k)
    Var_y_vec[k] = Var_y(Pm)

# Find k values for min and max variance of y(k)
k_var_max = np.argmax(Var_y_vec)
k_var_min = np.argmin(Var_y_vec)

print(
    'Variance of y(k) at each timestep: '
    + repr(Var_y_vec)
    )

print(
    'Answer to (b): We have the least knowledge about y(k) at timestep: '
    + repr(k_var_max)
    + ' and we have the most knowledge about y(k) at timestep: '
    + repr(k_var_min)
    )

''' intuitively it makes sense that we have the most knowledge of y(k) at k = 1
 since we are given a deterministic measurement of z(1) = 2.22 '''