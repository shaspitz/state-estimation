'''
Created on Mar 18, 2020

@author: Shawn Marshall-Spitzbart

UC Berkeley ME231B Assignment #5 Problem 2
'''

import numpy as np
import matplotlib.pyplot as plt

''' This problem involves the implementation of a time-varying Kalman filter
 for a system with Gaussian process and measurement noises '''

# First define global vars

# Time-invariant linear system as given in problem statement
N = 2
A = np.array([[0.8, 0.6], [-0.6, 0.8]])
H = np.array([[1, 0]])

# Time-invariant process and measurement noise covariances given as identity
V, W = np.eye(N), np.eye(1)

# Process and measurement noise are zero mean, gaussian, and independant

'''Function that returns value corresponding to Gaussian dist matching
 required mean/variance for process and measurement noises.
 Note that mean = 0 and var = I for both v(k) and w(k)'''


def r_normal(Ex, Var): return np.random.normal(Ex, Var, 1)

# Define measurement and time update for Kalman Filter implementation


def time_update(xm, Pm):
    xp = A @ xm
    Pp = A @ Pm @ A.transpose() + V
    return xp, Pp


def meas_update(xp, Pp, z):
    K = Pp @ H.transpose() @ np.linalg.inv(H @ Pp @ H.transpose() + W)
    xm = xp + K @ (z - H @ xp)
    Pm = (np.eye(N) - K @ H) @ Pp @ (np.eye(N) - K @ H).transpose() + K @ W @ K.transpose()
    return xm, Pm


# Define system propagation


def sym_sys(x_true):
    # Expecation and Var of v_k, w_k = 0 and I respectively
    v_k = np.matrix([r_normal(0, 1), r_normal(0, 1)])  # Process noise
    w_k = np.matrix([r_normal(0, 1)])  # Scalar measurement noise
    x_true = A*x_true + v_k
    z = H*x_true + w_k
    return x_true, z


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


def E_y(xm): return np.matrix([[1, 1]])*xm


def Var_y(Pm): return np.matrix([[1, 1]])*Pm*np.matrix([[1, 1]]).transpose()


E_y_val = E_y(xm)
Var_y_val = Var_y(Pm)

print(
    'Answer to (a): y(1) is normally distributed with mean: '
    + repr(round(E_y_val[0, 0], 4))
    + ' and variance: '
    + repr(round(Var_y_val[0, 0], 4))
    )

T_f = 11  # Simulation Timesteps (0 included)

# Initialize variance storage array with first components at k = 0
Var_y_vec = np.zeros([T_f, 1])
Var_y_vec[0] = Var_y(Pm_0)

# Intialize xm, Pm
xm, Pm = xm_0, Pm_0

# Initalize true state
x_true = np.matrix([r_normal(0, 3), r_normal(0, 1)])

for k in range(1, T_f):  # Note that estimates for k = 0 is initilized above

    # Simulate system and measurement
    x_true, z = sym_sys(x_true)

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

print(
    'Answer to (c): Expectation is linear, so E[e] = E[x(k)] - E[x_est(k)]. '
    + 'Also, since E[x(0)] = 0 and E[v(k)] = 0, recursively the true'
    + 'state evolves so that E[x(k)] = 0 for all k. ie E[e] is: '
    + repr(0 - xm)
    + ' Var[e] is... '
    + repr(Pm)
    )

# Answer to (d)

sim_tot = 10000  # Number of simulations

# Intialize xm, Pm
xm, Pm = xm_0, Pm_0

# preallocate parameters
e = np.zeros([sim_tot, T_f, N])
x_est = np.zeros([sim_tot, T_f, N])  # zeros for initial x_est

for sim in range(0, sim_tot):

    # Initalize true state
    x_true = np.matrix([r_normal(0, 3), r_normal(0, 1)])

    e[0, :] = (x_true - xm).transpose()

    for k in range(1, T_f):  # Note that estimate for k = 0 is initialized above

        # Simulate system and measurement
        x_true, z = sym_sys(x_true)

        # Kalman filter estimation: time update
        xp, Pp = time_update(xm, Pm)

        # Kalman filter estimation: measurement update
        xm, Pm = meas_update(xp, Pp, z)

        # Store results for plotting
        x_est[sim, k, :] = xm.transpose()
        e[sim, k, :] = (x_true - xm).transpose()

#  Plotting (d)
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
