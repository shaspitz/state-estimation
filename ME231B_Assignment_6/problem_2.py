'''
Created on Mar 24, 2020

@author: Shawn Marshall-Spitzbart

UC Berkeley ME231B Assignment #6 Problem 2
'''

import numpy as np
import scipy.linalg as sp
import sys

''' Compute the steady-state Kalman filter gain Kinf, for the given system '''

# Define global vars

# Time-invariant scalar linear system as given in problem statement
N = 1
A = 1
H = 1

# Process and measurement noise covariances
V, W = 1.0, 6.0

# Process and measurement noise are zero mean, gaussian, and independant

Pinf = sp.solve_discrete_are(A, H, V, W)  # No transpose on scalar A, H
Kinf = Pinf * H * (H * Pinf * H + W) ** -1

print('The steady-state Kalman filter gain, Kinf is: '
      + repr(round(Kinf[0, 0], 4)))

''' The true system dynamics differ from the ones used to derive the estimator
and are driven by parameter delta. For what parameters delta will the error
e(k) remain bounded as k tends to infinity?'''

# to test this, we run our KF and analyze e(k) for varying values of delta

'''Function that returns value corresponding to Gaussian dist matching
 required mean/variance for process and measurement noises.'''


def r_normal(Ex, Var): return np.random.normal(Ex, Var, 1)

# Define scalar measurement and time update for Kalman filter implementation


def time_update(xm, Pm):
    xp = A*xm
    Pp = A*Pm*A + V
    return xp, Pp


def meas_update(xp, Pp, z):
    K = Pp*H*(H*Pp*H + W)**-1
    xm = xp + K*(z - H*xp)
    Pm = (np.eye(N) - K*H)*Pp*(np.eye(N) - K*H) + K*W*K
    return xm, Pm


# Define simulation of true system dynamics


def sym_sys(x_true, delta):
    v_k = r_normal(0, V)  # Process noise
    w_k = r_normal(0, W)  # measurement noise
    x_true = (1 + delta) * x_true + v_k
    z = H * x_true + w_k
    return x_true, z


# Initialize estimate and covariance of state (at k = 0)
xm, Pm = 0.0, np.array([[3]])

# Initialize x_true at k = 0
x_true = r_normal(0, 3)

delta_list = np.array([-10, -2.01, -2.0, -1.0,
                       -0.01, 0, 0.01, 1, 10], dtype=float)
T_f = 100000  # Simulation Timesteps (0 included)
e_inf = np.zeros(len(delta_list))

max_float = sys.float_info.max/1000000  # Prevent overflow on next step

for i, config in np.ndenumerate(delta_list):

    for k in range(1, T_f):  # Note that k=0 is initialized above

        if xm < sys.float_info.max/1000000:  # prevent stack overflow

            # Simulate system and measurement
            x_true, z = sym_sys(x_true, config)

            # Kalman filter estimation: time update
            xp, Pp = time_update(xm, Pm)

            # Kalman filter estimation: measurement update
            xm, Pm = meas_update(xp, Pp, z)

            # Store e(10,000)
            e_inf[i] = x_true - xm

        else:
            e_inf[i] = float("inf")  # representing an unbounded error

    # Initialize estimate and covariance of state (at k = 0) for next config
    xm, Pm = 0.0, np.array([[3]])

    # Initialize x_true at k = 0 for next config
    x_true = r_normal(0, 3)

print('Our maximum of: ' + repr(round(sys.float_info.max/1000000, 4)) +
      ' is sufficiently large to approximate unboundedness.')

for i, val in np.ndenumerate(e_inf):

    if val != float("inf"):

        print('A delta value of ' + repr(round(delta_list[i], 4)) +
              ' produced a bounded error as k approaches infinity.')

print('From this we can deduce that e(k) remains bounded for delta in [-2, 0].')