'''
Created on Mar 23, 2020

@author: Shawn Marshall-Spitzbart

UC Berkeley ME231B Assignment #6 Problem 1
'''

import numpy as np
import scipy.linalg as sp

'''
(a) Using the standard Kalman filter, for various k values, what
is the variance of the posterior estimation error?
'''

# Define global vars

# Time-invariant scalar linear system as given in problem statement
N = 1
A = 1.2
H = 1

# Process and measurement noise covariances
V, W = 2, 1

# Process and measurement noise are zero mean, gaussian, and independant

'''
Function that returns value corresponding to Gaussian dist matching
required mean/variance for process and measurement noises.
Note that mean = 0 and var = I for both v(k) and w(k)
'''


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


def sym_sys(x_true):
    v_k = r_normal(0, V)  # Process noise
    w_k = r_normal(0, W)  # measurement noise
    x_true = A * x_true + v_k
    z = H * x_true + w_k
    return x_true, z


# Initialize estimate and covariance of state (at k = 0)
xm, Pm = 0, np.array([[3]])

# Initialize x_true at k = 0
x_true = r_normal(0, 3)

T_f = [2, 3, 11, 1001]  # Simulation Timesteps (0 included)
Var_est = np.zeros((4, 1))

for config in T_f:

    for k in range(1, config):  # Note that estimate for k=0 is initialized above

        # Simulate system and measurement
        x_true, z = sym_sys(x_true)

        # Kalman filter estimation: time update
        xp, Pp = time_update(xm, Pm)

        # Kalman filter estimation: measurement update
        xm, Pm = meas_update(xp, Pp, z)

    # Print variance of posterior estimation error (Pm(k)) for each config
    print('Variance of posterior estimation error for k = '
          + repr(k) + ' is: ' + repr(round(Pm[0, 0], 4)))


'''
(b) What is the steady-state Kalman filter for this system? Provide the
gain, and the steady-state posterior variance.
'''

Pinf = sp.solve_discrete_are(A, H, V, W)  # This is Pp as k -> inf
Kinf = Pinf * H * (H * Pinf * H + W) ** -1

'''
Use time update to transform Pinf (Pp as k -> inf)
into KF posterior variance (Pm as k -> inf)

Note: Pp converges to Pinf as k -> inf. Since K(k) also converges to to Kinf,
Pm must converge to our SSKF posterior variance using a measurement update.
'''
Pm_inf = (np.eye(N) - Kinf*H)*Pinf*(np.eye(N) - Kinf*H) + Kinf*W*Kinf

print('Steady state KF gain is: ' + repr(round(Kinf[0, 0], 4)))
print('Steady state KF posterior variance is: ' + repr(round(Pm_inf[0, 0], 4)))


'''
(c) Using now the steady-state Kalman filter to estimate the state,
at various values for k, what is the variance of the posterior estimation
 error? Comment on how this compares to the standard Kalman filter of part a.
'''

# Define scalar measurement update for SSKF implementation
# Note that this functions is equivalent to above with SSKF gain


def meas_update_ss(xp, Pp, z):
    xm = xp + Kinf*(z - H*xp)  # equivalent to A * xp + Kinf * z (from notes)
    Pm = (np.eye(N) - Kinf*H)*Pp*(np.eye(N) - Kinf*H) + Kinf*W*Kinf
    return xm, Pm


# Initialize estimate and covariance of state (at k = 0)
xm, Pm = 0, np.array([[3]])

# Initialize x_true at k = 0
x_true = r_normal(0, 3)

T_f = [2, 3, 11, 1001]  # Simulation Timesteps (0 included)
Var_est = np.zeros((4, 1))

for config in T_f:

    for k in range(1, config):  # Note that estimate for k=0 is initialized above

        # Simulate system and measurement
        x_true, z = sym_sys(x_true)

        # Kalman filter estimation: time update
        xp, Pp = time_update(xm, Pm)

        # Kalman filter estimation: measurement update
        xm, Pm = meas_update_ss(xp, Pp, z)

    # Print variance of posterior estimation error (Pm(k)) for each config
    print('Variance of posterior estimation error for k = '
          + repr(k) + ' is: ' + repr(round(Pm[0, 0], 4)))

print('The steady state Kalman Filter had slightly worse performance (higher'
      ' variance of posterior estimation error) than the standard Kalman'
      ' filter for k = 1 and 2. By k = 10 and 1000, both filters performed'
      ' satisfactorily with Pm ~= Pm_inf ')

