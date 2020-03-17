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

# Time-invariant process and measurement noises covariance given as identity matrix
global V, W
V, W = np.eye(N), np.eye(1)

# Mean of process noise and measurement noise given as zero
E_v, E_w = 0, 0

''' Function that returns value corresponding to uniform dist matching
 required mean/variance for process and measurement noises '''
r_uniform = lambda: np.random.uniform(-np.sqrt(3), np.sqrt(3))

# Define measurement and time update for Kalman Filter implementation


def meas_update(xp, Pp, z):
    K = Pp*np.transpose(H)*np.linalg.inv(H*Pp*np.transpose(H) + W)
    xm = xp + K*(z - H*xp)
    Pm = (np.eye(N) - K*H)*Pp*np.transpose(np.eye(N) - K*H) + K*W*np.transpose(K)
    return xm, Pm


def time_update(xm, Pm):
    xp = A*xm
    Pp = A*Pm*np.transpose(A) + V
    return xp, Pp


# Initialize prior estimate and prior covariance of state (at k = 0)
xm, Pm = np.zeros([2, 1]), np.array([[3, 0], [0, 1]])

T_f = 11  # Simulation Timesteps (0 included)
sim_tot = 10000  # Number of simulations

# preallocate parameters
e = np.zeros([sim_tot, T_f, N])  # there will be blank column 0
x_est = np.zeros([sim_tot, T_f, N])  # zeros for initial x_est

for sim in range(0, sim_tot):

    for k in range(1, T_f):  # Note that estimate for k=0 is initilized above

        # Simulate system and measurement
        x_true = A*xm + np.matrix([r_uniform(), r_uniform()]).transpose()  # Process noise here
        z = H*x_true + np.matrix([r_uniform()])  # Scalar measurement noise here

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
