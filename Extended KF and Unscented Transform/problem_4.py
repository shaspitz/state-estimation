'''
Created on Mar 29, 2020

@author: Shawn Marshall-Spitzbart

UC Berkeley ME231B Assignment #7 Problem 4
'''

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

'''
A battery's state of charge at time step k is
given by q(k), with q(k) = 1 corresponding to fully
charged and q(k) = 0 depleted. In each time step
(e.g. each hour) k, the battery powers a process
which consumes energy j(k) = j0 + v(k), where j0
is the average amount of energy consumed, and v(k)
is a random deviation, so that
q(k) = q(k-1) - j(k-1).
We have perfect knowledge of the battery's initial
charge, q(0) = 1, and we know that v(k) is white
and normally distributed as given in problem.

we now add a voltage sensor which gives us a noisy
reading of the battery voltage after each time cycle.
The measurement is z(k) = h(q(k)) + w(k) with w(k) normally distributed,
and h(q) is a nonlinear function mapping from current state of charge
to voltage, h(q) = 4 + (q - 1)**3
'''

# (a) Design an extended Kalman filter (EKF) to estimate the state of charge.


'''
Define given system and nonlinear measurement model. Note that process noise
will be set to zero for EKF but is included in function to abide by math
notation. Assume no inputs to system.
'''


def q_sys(q, vk): return q - (j0 + vk)


# Meas noise is included in 'h' defintion here to abide by EKF notation
def h_nlmeas(q, wk): return 4 + (q - 1)**3 + wk


# Define global vars
V, W, j0 = 0.05**2, 0.1**2, 0.1

# EKF initialization, q(0) = 1 with no uncertainty
xm, Pm = 1, 0


# Prior update: Jacobian matrices A and L:

'''
Scalar system model is already linear in q and v, therefore A and L jacobians
are constant for all k and equal to 1.
'''
Ak = 1
Lk = 1

# Define linear KF prediction function


def time_update(xm, Pm):
    xp = q_sys(xm, 0)
    Pp = Ak * Pm * Ak + Lk * V * Lk
    return xp, Pp


# Measurement update: Jacobian matrices H and M
q, w = sp.symbols('q w')

# Meas noise is included in 'h' defintion here to abide by EKF notation
h = 4 + (q - 1)**3 + w

# Take jacobians of nl meas model and lambdify to functions of np arrays
H = sp.lambdify([q, w], h.diff(q))
M = sp.lambdify([q, w], h.diff(w))

# Define linear KF measurement update function


def meas_update(xp, Pp, z):

    # Plug in values for H, L
    Hk = H(xp, 0)
    Mk = M(xp, 0)

    # KF equations
    Kk = Pp * Hk * (Hk * Pp * Hk + Mk * W * Mk)**-1
    xm = xp + Kk * (z - h_nlmeas(xp, 0))
    Pm = (np.eye(1) - Kk * Hk) * Pp

    return xm, Pm


'''
(c) Using the variance metric derived in (b), (see handwritten notes), make
a plot of the usefulness of a voltage measurement as a function of the
estimated state of charge, for q in [0 1] (with usefulness as defined in
the subproblem b). Set Pp(k) = 0.1, and W = 0.1. Where is the measurement
most informative? Where is it least informative?
'''
q, Pp_c, W_c = np.linspace(0, 1, 100), 0.1, 0.1


# note that 'H' in this problem is same as lambda function 'H' defined above
usefulness = H(q, 0)**2 * Pp_c / (W + H(q, 0)**2 * Pp_c)

plt.figure(0)
plt.plot(q, usefulness, linewidth=3)
plt.xlabel('Estimated State of Charge')
plt.ylabel('Usefulness of a Voltage Measurement')
plt.title(r'Usefulness of a Voltage Measurement vs Estimated State of Charge')

print('The measurements are most informative for low state of charge values,'
      ' and least informative for high state of charge values. This makes'
      ' sense intuitively because the battery starts at a state of charge'
      ' of q = 1 with no uncertainty. After each succsessive timestep, another'
      ' process uncertainty value, v(k), is imposed on the system, thus'
      ' increasing the cumulative process unceratinty of our battery state.'
      ' Measurements used at this area of high process uncertainty (low q)'
      ' are most useful because the EKF is an optimal estimator, and will'
      ' weight measurement values higher when the process model'
      ' is less reliable.')


'''
For the remainder of the problem, use the following sequence of measurements
starting at time k = 1 (dummy meas added at k = 0 to abide by math notiation):
'''
z = np.array([0, 4.04, 3.81, 3.95, 3.90, 3.88, 3.88, 3.90, 3.55, 3.18])

'''
(d) Run your extended Kalman filter with this data, and generate two plots:
the estimated state of charge, and the variance of this estimate, across k.
'''

Tf = 9

# Initialze plotting arrays
q_est, var_est = np.zeros(10), np.zeros(10)
q_est[0], var_est[0] = xm, Pm

# Note that EKF is already initialized for k = 0 in (a)
for k in range(1, Tf + 1):

    # Run EKF
    xp, Pp = time_update(xm, Pm)
    xm, Pm = meas_update(xp, Pp, z[k])

    # Store values for plotting
    q_est[k], var_est[k] = xm, Pm

#  Plotting
plt.figure(1)
plt.plot(range(0, Tf + 1), q_est, linewidth=3)
plt.xlabel('Timestep (k)')
plt.ylabel('Estimated State of Charge (Using Measurements)')
plt.title(r'Estimated State of Charge vs Time (Using Measurements)')

plt.figure(2)
plt.plot(range(0, Tf + 1), var_est, linewidth=3)
plt.xlabel('Timestep (k)')
plt.ylabel('Variance of Estimated State of Charge (Using Measurements)')
plt.title(r'Variance of Estimated State of Charge vs Time (Using Measurements)')

'''
(e) After 9 steps, what would the mean and variance be if you did not have the
voltage measurements. How does this compare to your EKF output?
'''
# Initialze plotting arrays
q_est_nodata, var_est_nodata = np.zeros(10), np.zeros(10)
q_est_nodata[0], var_est_nodata[0] = 1, 0

# Intialize xp, Pp for EKF at k = 0
xp, Pp = 1, 0

for k in range(1, Tf + 1):

    # Run EKF
    xp, Pp = time_update(xp, Pp)

    # Store values for plotting
    q_est_nodata[k], var_est_nodata[k] = xp, Pp

print(
    'After 9 steps, the mean and variance without using voltage measurements'
    ' would be: ' + repr(round(q_est_nodata[Tf], 4)) + ' and: '
    + repr(round(var_est_nodata[Tf], 4)) + ' respectively.')

print(
    'After 9 steps, the mean and variance for our EKF estimate'
    ' would be: ' + repr(round(q_est[Tf], 4)) + ' and: '
    + repr(round(var_est[Tf], 4)) + ' respectively.')

print('By comparing our EKF output to a system estimation that doesn\'t'
      ' use measurements, we can see that by timestep k = 9, the EKF is able'
      ' to keep the estimate variance significantly lower than that of an'
      ' estimate without measurements. The EKF also gave us an estimate that'
      ' was slightly lower than our no data model.'
      )

plt.show()
