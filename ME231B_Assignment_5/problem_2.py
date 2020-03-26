'''
Created on Mar 18, 2020

@author: Shawn Marshall-Spitzbart

UC Berkeley ME231B Assignment #5 Problem 2
'''

import numpy as np
import matplotlib.pyplot as plt

''' This problem involves the implementation of a time-varying Kalman filter
 for a system with Gaussian process and measurement noises '''

# (a) Compute the PDF of y(1) given the observation z(1) = 2:22

# Define global vars

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
    Pp = A @ Pm @ A.T + V
    return xp, Pp


def meas_update(xp, Pp, z):
    K = Pp @ H.T @ np.linalg.inv(H @ Pp @ H.T + W)
    xm = xp + K @ (z - H @ xp)
    Pm = (np.eye(N) - K @ H) @ Pp @ (np.eye(N) - K @ H).T + K @ W @ K.T
    return xm, Pm


# Define system propagation


def sym_sys(x_true):
    # Expecation and Var of v_k, w_k = 0 and I respectively
    v_k = np.array([[r_normal(0, 1)[0], r_normal(0, 1)[0]]]).T  # Process noise
    w_k = np.array([[r_normal(0, 1)[0]]])  # Scalar measurement noise
    x_true = A @ x_true + v_k
    z = H @ x_true + w_k
    return x_true, z


# Initialize estimate and covariance of state (at k = 0)
xm_0, Pm_0 = np.zeros((2, 1)), np.array([[3, 0], [0, 1]])

# (a) Compute the PDF of y(1) given the observation z(1) = 2.22
# First simulate prediction and measurement update steps of KF forward to k = 1
xp, Pp = time_update(xm_0, Pm_0)
xm, Pm = meas_update(xp, Pp, 2.22)

'''
Since output y(k) is an affine linear transformation of GRV x(k), y(k) itself
is a GRV with mean [1 1]*xm(k) and variance [1 1]*Pm(k)*[1 1]' (see property 1
of GRVs from chapter 8 class notes)
'''


def E_y(xm): return np.array([[1, 1]]) @ xm


def Var_y(Pm): return np.array([[1, 1]]) @ Pm @ np.array([[1, 1]]).T


E_y_val = E_y(xm)
Var_y_val = Var_y(Pm)

print(
    'y(1) is normally distributed with mean: '
    + repr(round(E_y_val[0, 0], 4))
    + ' and variance: '
    + repr(round(Var_y_val[0, 0], 4))
    )

'''
(b) At which time step k (for k <= 10) do you have least knowledge about y(k)?
For which do you have the most knowledge? Interpret "little knowledge of
something" as "our estimate of something has large variance".
'''

T_f = 11  # Simulation Timesteps (0 included)

# Initialize variance storage array with first components at k = 0
Var_y_vec = np.zeros(T_f)
Var_y_vec[0] = np.diagonal(Var_y(Pm_0))

# Intialize xm, Pm
xm, Pm = xm_0, Pm_0

# Initalize true state
x_true = np.array([[r_normal(0, 3)[0], r_normal(0, 1)[0]]]).T

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

for i, val in np.ndenumerate(Var_y_vec):
    print(
        'Variance of y(k) at timestep: ' + repr(i[0]) + ' is: '
        + repr(round(val, 4)))

print(
    'We have the least knowledge about y(k) at timestep: '
    + repr(k_var_max)
    + ' and we have the most knowledge about y(k) at timestep: '
    + repr(k_var_min)
    )

# (c) Compute the PDF of e(10)

E_e10_part_c = np.array([[0], [0]])
Var_e10_part_c = Pm

print(
    'With all random variables having expecations of zero,'
    ' and the KF being initialized with E[x0], it follows that the filter,'
    ' is unbiased, ie E[e] is: '
    + repr(E_e10_part_c)
    + ' and Var[e(k)] = E[(x(k) - xm(k))(x(k) - xm(k)).T] = Pm(k) =: '
    + repr(Var_e10_part_c)
    )

'''
(d) Implement a simulation of the system and a Kalman filter that produces an
estimate x(k). Execute this 10,000 times and plot a histogram of the resulting
values for e(10) (make two histograms,one for each component of e(10)).
 Comment on how this compares to your answer in c).
'''

sim_tot = 10000  # Number of simulations

# Intialize xm, Pm
xm, Pm = xm_0, Pm_0

# preallocate parameters
e = np.zeros([sim_tot, T_f, N])
x_est = np.zeros([sim_tot, T_f, N])  # zeros for initial x_est

for sim in range(0, sim_tot):

    # Initalize true state
    x_true = np.array([[r_normal(0, 3)[0], r_normal(0, 1)[0]]]).T

    e[sim, 0, :] = (x_true - xm).ravel()

    for k in range(1, T_f):  # Note that estimate for k = 0 is initialized above

        # Simulate system and measurement
        x_true, z = sym_sys(x_true)

        # Kalman filter estimation: time update
        xp, Pp = time_update(xm, Pm)

        # Kalman filter estimation: measurement update
        xm, Pm = meas_update(xp, Pp, z)

        # Store results for plotting
        x_est[sim, k, :] = xm.ravel()
        e[sim, k, :] = (x_true - xm).ravel()

#  Plots
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

'''
To compare the results from part (c) and part (d),
we compute the mean and variance of each e(10) component over all the
simulations.
'''

E_e_10_part_d = np.mean(e[:, 10, :], axis=0)
Var_e_10_part_d = np.var(e[:, 10, :], axis=0)

print(' The expectations for each component of e(10) over the 10,000 '
      'simulations from part (d) are: ' + repr(round(E_e_10_part_d[0], 4)) +
      ' and ' + repr(round(E_e_10_part_d[1], 4)) + ' respectivelty. Both of '
      'these values are relatively close to the values from part (c)'
      ' of: ' + repr(round(E_e10_part_c[0, 0], 4)) + ' and '
      + repr(E_e10_part_c[1, 0])
      )

print(' The variances for each component of e(10) over the 10,000 '
      'simulations from part (d) are: ' + repr(round(Var_e_10_part_d[0], 4))
      + ' and ' + repr(round(Var_e_10_part_d[1], 4)) + ' respectivelty. Both '
      'of these values are relatively close to the values from part (c)'
      ' of: ' + repr(round(Var_e10_part_c[0, 0], 4)) + ' and '
      + repr(round(Var_e10_part_c[1, 1], 4))
      )

plt.show()
