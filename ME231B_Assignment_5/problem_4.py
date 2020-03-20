'''
Created on Mar 13, 2020

@author: Shawn Marshall-Spitzbart

UC Berkeley ME231B Assignment #5 Problem 4
'''

import numpy as np
import matplotlib.pyplot as plt

''' This problem involves the implementation of a time-varying Kalman filter
 for a system with uniform (ie non-Gaussian) process and measurement noises '''

# Time-invariant linear system as given in problem statement
global N, A, H
N = 2
A = np.matrix([[0.8, 0.6], [-0.6, 0.8]])
H = np.matrix([1, 0])

# Time-invariant process and measurement noise covariances given as identity
global V, W
V, W = np.eye(N), np.eye(1)

# Process and measurement noise are zero mean, white, and independant

''' Function that returns value corresponding to uniform dist matching
 required mean/variance for process and measurement noises '''


def r_uniform(a, b): return np.random.uniform(a, b)

# Define measurement and time update for Kalman Filter implementation


def time_update(xm, Pm):
    xp = A*xm
    Pp = A*Pm*A.transpose() + V
    return xp, Pp


def meas_update(xp, Pp, z):
    K = Pp*H.transpose()*np.linalg.inv(H*Pp*H.transpose() + W)
    xm = xp + K*(z - H*xp)
    Pm = (np.eye(N) - K*H)*Pp*np.transpose(np.eye(N) - K*H) + K*W*K.transpose()
    return xm, Pm


# Define system propagation


def sym_sys(x_true):
    # a = -sqrt(3), b = sqrt(3) for uniform dist of v_k and w_k (see writing)
    v_k = np.matrix([r_uniform(-np.sqrt(3), np.sqrt(3)), r_uniform(-np.sqrt(3),
        np.sqrt(3))]).transpose()  # Process noise
    w_k = np.matrix([r_uniform(-np.sqrt(3), np.sqrt(3))])  # measurement noise
    x_true = A*x_true + v_k
    z = H*x_true + w_k
    return x_true, z


# Initialize estimate and covariance of state (at k = 0)
xm, Pm = np.zeros([2, 1]), np.matrix([[3, 0], [0, 1]])

T_f = 11  # Simulation Timesteps (0 included)
sim_tot = 10000  # Number of simulations

# preallocate parameters
e = np.zeros([sim_tot, T_f, N])
x_est = np.zeros([sim_tot, T_f, N])  # zeros for initial x_est

for sim in range(0, sim_tot):

    # Initalize true state, a = -3, b = 3 for uniform dist of x[0] (see writing)
    x_true = np.matrix([r_uniform(-3, 3), r_uniform(-3, 3)]).transpose()
    e[sim, 0, :] = (x_true - xm).transpose()

    for k in range(1, T_f):  # Note that estimate for k=0 is initialized above

        # Simulate system and measurement
        x_true, z = sym_sys(x_true)

        # Kalman filter estimation: time update
        xp, Pp = time_update(xm, Pm)

        # Kalman filter estimation: measurement update
        xm, Pm = meas_update(xp, Pp, z)

        # Store results for plotting
        x_est[sim, k, :] = xm.transpose()
        e[sim, k, :] = (x_true - xm).transpose()

#  Display ensemble average and variance for each component of e(10) and estimator output
print('Ensemble Average of First component for e(10): ' + repr(round(np.mean(e[:, 10, 0]), 4)))
print('Ensemble Variance of First component for e(10): ' + repr(round(np.var(e[:, 10, 0]), 4)))
print('Ensemble Average of First component for Estimator Output at Timestep 10: '
      + repr(round(np.mean(x_est[:, 10, 0]), 4)) + '\n')

print('Ensemble Average of Second component for e(10): ' + repr(round(np.mean(e[:, 10, 1]), 4)))
print('Ensemble Variance of Second component for e(10): ' + repr(round(np.var(e[:, 10, 1]), 4)))
print('Ensemble Average of Second component for Estimator Output at Timestep 10: '
      + repr(round(np.mean(x_est[:, 10, 1]), 4)))

#  Plotting
num_bins = 20

plt.figure(0)
plt.hist(e[:, 10, 0], num_bins, facecolor='blue', edgecolor='black', alpha=1)
plt.xlabel('First Component For Error e at timestep 10')
plt.ylabel('Count')
plt.title(r'Histogram of First Component for Error (e) at timestep 10')

plt.figure(1)
plt.hist(e[:, 10, 1], num_bins, facecolor='blue', edgecolor='black', alpha=1)
plt.xlabel('Second Component For Error e at timestep 10')
plt.ylabel('Count')
plt.title(r'Histogram of Second Component for Error (e) at timestep 10')

plt.show()
