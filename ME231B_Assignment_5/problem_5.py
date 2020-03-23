'''
Created on Mar 19, 2020

@author: Shawn Marshall-Spitzbart

UC Berkeley ME231B Assignment #5 Problem 4
'''

import numpy as np
import matplotlib.pyplot as plt

''' In this problem we estimate the behavior of a
city's water network, where fresh water is supplied by
a desalination plant, and consumed at various points
in the network. The network contains various con-
sumers, and various reservoirs where water is locally
stored.
Our objective is to use noisy information about pro-
duction and consumption levels, and noisy measure-
ments of the amount of water in the reservoirs, to
estimate the state of the network. We will consider 3
networks of increasing complexity. '''

# Part (a)

# First define global vars

# Time-invariant linear system deduced from problem statement
N = 1
A = 1
H = 1

# Time-invariant process and sensor uncertainity (variance)
V, W = 9, 25

# Process and measurement noise are zero mean, white, and independant

# define constant supply of water, d, and predicted customer use, m
d, m = 10, 7

# Define measurement and time update for Kalman filter implementation


def time_update_a(xm, Pm):
    xp = A*xm - m + d  # state update constants are appropriately added here
    Pp = A*Pm*A + V
    return xp, Pp


def meas_update_a(xp, Pp, z):
    K = Pp*H*(H*Pp*H + W)**-1
    xm = xp + K*(z - H*xp)
    Pm = (np.eye(N) - K*H)*Pp*(np.eye(N) - K*H) + K*W*K
    return xm, Pm


# Initialize estimate and covariance of state (at k = 0)
xm, Pm = 20, 25

T_f = 11  # Simulation Timesteps (0 included)

# preallocate parameters
x_est = np.zeros([T_f])
x_est[0] = xm
P_est = np.zeros([T_f])
P_est[0] = Pm

# Given measurements
z = np.array([32.1, 39.8, 45.9, 52.0, 51.0, 63.4, 58.1, 60.3, 76.6, 73.0])

for k in range(1, T_f):  # Note that estimate for k=0 is initialized above

    # Kalman filter estimation: time update
    xp, Pp = time_update_a(xm, Pm)

    # Kalman filter estimation: measurement update
    xm, Pm = meas_update_a(xp, Pp, z[k-1])

    # Store results for plotting
    x_est[k] = xm
    P_est[k] = Pm

#  Plotting
num_bins = 20

plt.figure(0)
plt.plot(range(0, T_f), x_est)
plt.xlabel('Timestep (k)')
plt.ylabel('Estimated Volume of Water, r')
plt.title(r'Estimated Volume of Water, r vs Time (Part a)')

plt.figure(1)
plt.plot(range(0, T_f), P_est)
plt.xlabel('Timestep (k)')
plt.ylabel('Uncertainty of Estimated Volume of Water, r')
plt.title(r'Uncertainty of Estimated Volume of Water, r vs Time (Part a)')

# Part (b)

# First define global vars

# Time-invariant linear system deduced from problem statement
N = 2
A = np.array([[1, -1], [0, 1]])
H = np.array([[1, 0]])

# Time-invariant process and sensor uncertainity (variance)
V, W = 0.1, 25

# Process and measurement noise are zero mean, white, and independant

# Constant supply of water, d, is defined above

# Define measurement and time update for Kalman filter implementation


def time_update_b(xm, Pm):
    # Known input, d added to system
    xp = A @ xm + np.array([[d], [0]])
    # Process noise only affects second component of state below
    Pp = A @ Pm @ A.transpose() + np.array([[0], [V]])
    return xp, Pp


def meas_update_b(xp, Pp, z):
    K = Pp @ H.transpose() @ np.linalg.inv(H @ Pp @ H.transpose() + W)
    xm = xp + K @ (z - H @ xp)
    Pm = (np.eye(N) - K @ H) @ Pp @ np.transpose(np.eye(N) - K @ H)
    + K * W * K.transpose()
    return xm, Pm


# Initialize estimate and covariance of state (at k = 0)
xm, Pm = np.array([[20], [7]]), np.array([[25, 0], [0, 1]])

T_f = 11  # Simulation Timesteps (0 included)

# preallocate parameters
x_est = np.zeros((T_f, 2))
x_est[0] = xm.ravel()
P_est = np.zeros((T_f, 4))
P_est[0] = Pm.ravel()

# Given measurements are same as part (a)

for k in range(1, T_f):  # Note that estimate for k=0 is initialized above

    # Kalman filter estimation: time update
    xp, Pp = time_update_b(xm, Pm)

    # Kalman filter estimation: measurement update
    xm, Pm = meas_update_b(xp, Pp, z[k-1])

    # Store results for plotting
    x_est[k] = xm.ravel()
    P_est[k] = Pm.ravel()

#  Plotting
num_bins = 20

plt.figure(2)
plt.plot(range(0, T_f), x_est[:, 0])
plt.xlabel('Timestep (k)')
plt.ylabel('Estimated Volume of Water, r')
plt.title(r'Estimated Volume of Water, r vs Time (Part b)')

plt.figure(3)
plt.plot(range(0, T_f), x_est[:, 1])
plt.xlabel('Timestep (k)')
plt.ylabel('Estimated Consumption of Water, c')
plt.title(r'Estimated Consumption of Water, c vs Time (Part b)')

plt.figure(4)
plt.plot(range(0, T_f), P_est[:, 0])
plt.xlabel('Timestep (k)')
plt.ylabel('Uncertainty of Estimated Volume of Water, r')
plt.title(r'Uncertainty of Estimated Volume of Water, r vs Time (Part b)')

plt.figure(5)
plt.plot(range(0, T_f), P_est[:, 3])
plt.xlabel('Timestep (k)')
plt.ylabel('Uncertainty of Estimated Consumption of Water, c')
plt.title(r'Uncertainty of Estimated Consumption of Water, c vs Time (Part b)')

plt.show()

print(0)

