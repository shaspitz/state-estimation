'''
Created on Mar 13, 2020

@author: Shawn Marshall-Spitzbart

UC Berkeley ME231B Assignment #5 Problem 4
'''

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

'''
This problem involves the implementation of a time-varying Kalman filter
for a system with uniform (ie non-Gaussian) process and measurement noises
'''

'''
Design an optimal linear estimator for this system (KF), and
implement a simulation of the system and a Kalman filter that produces an
estimate as a function of k. For the simulation, draw all of the random
variables from uniform distributions that match the required mean/variance.
Execute this 10,000 times and plot a histogram of the resulting values for
e(10) (make two histograms, one for each component of e(10)). Comment on
how this compares to your answers in Problem 2c) and Problem 2d). Also give
the ensemble average and variance for each component of e(10)
(that is, the average over the samples from the simulations for each
component). Compare that to the estimator's output, and briefly discuss.
'''

# Define global vars

# Time-invariant linear system as given in problem statement
N = 2
A = np.array([[0.8, 0.6], [-0.6, 0.8]])
H = np.array([[1, 0]])

# Time-invariant process and measurement noise covariances given as identity
V, W = np.eye(N), np.eye(1)

# Process and measurement noise are zero mean, white, and independant

'''
The problem asks us to draw all of the random variables from uniform
distributions matching the required mean/variance. In order to implement
this in python, we need the bounds of each uniform distribution. The variance
for a uniform distribution is: Var = (b - a)^2 / 12 where 'b' is the upper
bound of the distribution and 'a' is the lower bound. The mean for a uniform
distribution is then Mean = (a + b) / 2. We will solve for these bounds
using Sympy and the given problem info for x0, w(k), and v(k).
'''


def get_bounds(mean, var):
    a, b = sp.Symbol("a"), sp.Symbol("b")
    eqtns = (sp.Eq((a + b) / 2, mean), sp.Eq((b - a)**2 / 12, var))
    ans = sp.solve(eqtns, (a, b))
    return ans[0][0], ans[0][1]  # return first answer, 'b' being larger


# mean and variance for each component of v(k) and w(k) is 0 and 1 respectively
mean_vw, var_vw = 0, 1
a_vw, b_vw = get_bounds(mean_vw, var_vw)

# mean and variance for 1st component of x(k) is 0 and 3 respectively
mean_x1, var_x1 = 0, 3
a_x1, b_x1 = get_bounds(mean_x1, var_x1)

# mean and variance for 1st component of x(k) is 0 and 1 respectively
mean_x2, var_x2 = 0, 1
a_x2, b_x2 = get_bounds(mean_x2, var_x2)

'''
Next define function that returns value corresponding to uniform dist
matching required mean/variance for process and measurement noises
'''


def r_uniform(a, b): return np.random.uniform(a, b)

# Define measurement and time update for Kalman Filter implementation


def time_update(xm, Pm):
    xp = A @ xm
    Pp = A @ Pm @ A.transpose() + V
    return xp, Pp


def meas_update(xp, Pp, z):
    K = Pp @ H.transpose() @ np.linalg.inv(H @ Pp @ H.transpose() + W)
    xm = xp + K @ (z - H @ xp)
    Pm = (np.eye(N) - K @ H) @ Pp @ (np.eye(N) - K @ H).transpose()
    + K @ W @ K.transpose()
    return xm, Pm


# Define system propagation


def sym_sys(x_true):
    # a = -sqrt(3), b = sqrt(3) for uniform dist of v_k and w_k (see writing)
    v_k = np.array([[r_uniform(a_vw, b_vw)],
                    [r_uniform(a_vw, b_vw)]])  # Process noise
    w_k = np.array([r_uniform(a_vw, b_vw)])  # measurement noise
    x_true = A @ x_true + v_k
    z = H @ x_true + w_k
    return x_true, z


# Initialize estimate and covariance of state (at k = 0)
xm, Pm = np.zeros((2, 1)), np.array([[3, 0], [0, 1]])

T_f = 11  # Simulation Timesteps (0 included)
sim_tot = 10000  # Number of simulations

# preallocate parameters
e = np.zeros((sim_tot, T_f, N))
x_est = np.zeros((sim_tot, T_f, N))  # zeros for initial x_est

for sim in range(0, sim_tot):

    # Initalize true state, a = -3, b = 3 for uniform dist of x[0]
    x_true = np.array([[r_uniform(a_x1, b_x1)], [r_uniform(a_x2, b_x2)]])
    e[sim, 0, :] = (x_true - xm).ravel()

    for k in range(1, T_f):  # Note that estimate for k=0 is initialized above

        # Simulate system and measurement
        x_true, z = sym_sys(x_true)

        # Kalman filter estimation: time update
        xp, Pp = time_update(xm, Pm)

        # Kalman filter estimation: measurement update
        xm, Pm = meas_update(xp, Pp, z)

        # Store results for plotting
        x_est[sim, k, :] = xm.ravel()
        e[sim, k, :] = (x_true - xm).ravel()

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

print(
    'The results from these plots are similar to problem (2c) and (2d) in that'
    ' the averages for each component of e(10) are approximately centered'
    ' around zero. Additionally, the variances for each component of e(10) '
    ' are relatively similar for both problems at ~ 0.7 and ~2.3'
    ' respectively. It makes sense that each problem formulation yeilds'
    ' similar results, since the KF doesnt require RVs to have any particular'
    ' distribution. '
    )

#  Display ensemble average, variance and est output for each comp of e(10)
print('Ensemble Average of First component for e(10): '
      + repr(round(np.mean(e[:, 10, 0]), 4)))
print('Ensemble Variance of First component for e(10): '
      + repr(round(np.var(e[:, 10, 0]), 4)))
print(
    'Ensemble Average of First component for Estimator Output at Timestep 10: '
    + repr(round(np.mean(x_est[:, 10, 0]), 4)) + '\n'
      )

print('Ensemble Average of Second component for e(10): '
      + repr(round(np.mean(e[:, 10, 1]), 4)))
print('Ensemble Variance of Second component for e(10): '
      + repr(round(np.var(e[:, 10, 1]), 4)))
print(
    'Ensemble Average of Second component for Estimator Output at Timestep 10: '
    + repr(round(np.mean(x_est[:, 10, 1]), 4))
      )

print(
    'The ensemble averages for each component of e(10) are relatively close to'
    ' zero, meaning that the estimator is approximately unbiased, which makes'
    ' sense. The variance of the second component for e(10) is higher than'
    ' that of the first component, meaning that the estimator is less'
    ' confident in its estimate of the second state.'
    )

plt.show()

