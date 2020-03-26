'''
Created on Mar 19, 2020

@author: Shawn Marshall-Spitzbart

UC Berkeley ME231B Assignment #5 Problem 4
'''

import numpy as np
import matplotlib.pyplot as plt

'''
In this problem we estimate the behavior of a
city's water network, where fresh water is supplied by
a desalination plant, and consumed at various points
in the network. The network contains various con-
sumers, and various reservoirs where water is locally
stored.
Our objective is to use noisy information about pro-
duction and consumption levels, and noisy measure-
ments of the amount of water in the reservoirs, to
estimate the state of the network. We will consider 3
networks of increasing complexity.
'''

'''
(a) We will first investigate a very simplified system, with a single
reservoir and single consumer (i.e. we only have one tank level r
to keep track of). We will model our consumers as c(k) = m + v(k),
where m is the typical consumption, and v(k) is a zero-mean uncertainty,
assumed white and independent of all other quantities. Using the Kalman filter,
estimate the actual volume of water r for various k. Also provide the associated
uncertainty for your estimate of r. Provide this using graphs.
'''

# Define global vars

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
plt.figure(0)
plt.plot(range(0, T_f), x_est, linewidth=3)
plt.xlabel('Timestep (k)')
plt.ylabel('Estimated Volume of Water')
plt.title(r'Estimated Volume of Water vs Time (Part a)')

plt.figure(1)
plt.plot(range(0, T_f), P_est, linewidth=3)
plt.xlabel('Timestep (k)')
plt.ylabel('Uncertainty of Estimated Volume of Water')
plt.title(r'Uncertainty of Estimated Volume of Water vs Time (Part a)')

'''
(b) We now extend the previous part by also estimating the consumption, c(k).
We model the consumption as evolving as c(k) = c(k-1) + n(k-1), where n(k-1)
is a zero-mean random variable, which is white and independent of all other
quantities. Using the Kalman filter, estimate the actual volume of water r for
various k, and the consumption rate c(k). Also provide the associated
uncertainty for both. Provide this using graphs. How do your answers for r
compare to the previous case?
'''

# Define global vars

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
    Pp = A @ Pm @ A.T + np.array([[0], [V]])
    return xp, Pp


def meas_update_b(xp, Pp, z):
    K = Pp @ H.T @ np.linalg.inv(H @ Pp @ H.T + W)
    xm = xp + K @ (z - H @ xp)
    Pm = (np.eye(N) - K @ H) @ Pp @ (np.eye(N) - K @ H).T
    + K * W * K.T
    return xm, Pm


# Initialize estimate and covariance of state (at k = 0)
xm, Pm = np.array([[20], [7]]), np.array([[25, 0], [0, 1]])

# preallocate parameters
x_est = np.zeros((T_f, N))
x_est[0] = xm.ravel()
P_est = np.zeros((T_f, N))
P_est[0] = Pm.diagonal()

# Given measurements are same as part (a)

for k in range(1, T_f):  # Note that estimate for k=0 is initialized above

    # Kalman filter estimation: time update
    xp, Pp = time_update_b(xm, Pm)

    # Kalman filter estimation: measurement update
    xm, Pm = meas_update_b(xp, Pp, z[k-1])

    # Store results for plotting
    x_est[k] = xm.ravel()
    P_est[k] = Pm.diagonal()

#  Plotting
plt.figure(2)
plt.plot(range(0, T_f), x_est[:, 0], linewidth=3)
plt.xlabel('Timestep (k)')
plt.ylabel('Estimated Volume of Water')
plt.title(r'Estimated Volume of Water vs Time (Part b)')

plt.figure(3)
plt.plot(range(0, T_f), x_est[:, 1], linewidth=3)
plt.xlabel('Timestep (k)')
plt.ylabel('Estimated Consumption of Water')
plt.title(r'Estimated Consumption of Water vs Time (Part b)')

plt.figure(4)
plt.plot(range(0, T_f), P_est[:, 0], linewidth=3)
plt.xlabel('Timestep (k)')
plt.ylabel('Uncertainty of Estimated Volume of Water')
plt.title(r'Uncertainty of Estimated Volume of Water vs Time (Part b)')

plt.figure(5)
plt.plot(range(0, T_f), P_est[:, 1], linewidth=3)
plt.xlabel('Timestep (k)')
plt.ylabel('Uncertainty of Estimated Consumption of Water')
plt.title(r'Uncertainty of Estimated Consumption of Water vs Time (Part b)')

print('The answers for r in part (a) and part (b) are shaped similarily.'
      ' However, the steady state uncertainty from (b) was lower than that'
      ' of part (a) as shown in the graphs.')

'''
(c) Now, we extend the system to a more complex network with four reservoirs,
and four consumers. Using the Kalman filter, estimate the actual volume of
water r(i) over various k for all tanks. Also provide the associated
uncertainty. Provide this using graphs. Notice that the estimate uncertainty
for tank 1 is lower than the solution you had for the simpler
network, why is this?
'''

# Define global vars

# Time-invariant linear system deduced from problem statement
N = 8
d = 30  # constant supply of water d(k)
alph = 0.3  # Flow balancing of 0.3

A = np.concatenate((
    np.concatenate((
        np.array([[1-2*alph, alph, alph, 0],
                  [alph, 1-2*alph, alph, 0],
                  [alph, alph, 1-2*alph, 0],
                  [0, 0, alph, 1-alph]]),
        -1*np.eye(4)), axis=1),
    np.concatenate((np.zeros((4, 4)), np.eye(4)), axis=1)),
    axis=0)

H = np.concatenate((np.eye(4), np.zeros((4, 4))), axis=1)

# Time-invariant process and measurement noise covariances
V = 0.1 * np.eye(N)
V[0:4, 0:4] = 0  # Adjust V so process noise only affects last four states
W = 25 * np.eye(4)  # z(k) is of size 4

# Process and measurement noise are zero mean, white, and independant

# Define measurement and time update for Kalman Filter implementation


def time_update_c(xm, Pm):
    xp = A @ xm
    Pp = A @ Pm @ A.T + V
    return xp, Pp


def meas_update_c(xp, Pp, z):
    K = Pp @ H.T @ np.linalg.inv(H @ Pp @ H.T + W)
    xm = xp + K @ (z - H @ xp)
    Pm = (np.eye(N) - K @ H) @ Pp @ (np.eye(N) - K @ H).T
    + K @ W @ K.T
    return xm, Pm


# Initialize estimate and covariance of state (at k = 0)
xm = np.array([[20, 40, 60, 20, 7, 7, 7, 7]]).T
Pm = np.diag([20, 20, 20, 20, 1, 1, 1, 1])

# preallocate parameters
x_est = np.zeros((T_f, N))
x_est[0] = xm.ravel()
P_est = np.zeros((T_f, N))
P_est[0] = Pm.diagonal()

# Given measurements
z = np.array([[62.6, 70.3, 73.5, 77.2, 73.2, 94.2, 87.4, 89.7, 90.4, 94.2],
              [29.4, 44.8, 37.3, 40.1, 44.1, 43.8, 53.8, 49.9, 51.9, 52.1],
              [35.9, 38.8, 25.9, 39.0, 31.2, 46.9, 39.6, 44.0, 42.5, 54.2],
              [40.9, 21.9, 18.0, 08.8, 23.9, 17.2, 18.6, 22.0, 22.9, 17.2]])

for k in range(1, T_f):  # Note that estimate for k=0 is initialized above

    # Kalman filter estimation: time update
    xp, Pp = time_update_c(xm, Pm)

    # Kalman filter estimation: measurement update
    # (ensuring z(k) is 4x1 after slicing)
    xm, Pm = meas_update_c(xp, Pp, z[0:4, k-1].reshape(4, 1))

    # Store results for plotting
    x_est[k] = xm.ravel()
    P_est[k] = Pm.diagonal()

#  Plotting
plt.figure(6)
plt.plot(range(0, T_f), x_est[:, 0:4], linewidth=3)
plt.xlabel('Timestep (k)')
plt.ylabel('Estimated Volume of Water')
plt.title(r'Estimated Volume of Water vs Time (Part c)')
plt.legend(labels=['Tank 1', 'Tank 2', 'Tank 3', 'Tank 4'], loc="upper center")

plt.figure(7)
plt.plot(range(0, T_f), x_est[:, 4:9], linewidth=3)
plt.xlabel('Timestep (k)')
plt.ylabel('Estimated Consumption of Water')
plt.title(r'Estimated Consumption of Water vs Time (Part c)')
plt.legend(labels=['Tank 1', 'Tank 2', 'Tank 3', 'Tank 4'], loc="lower left")

linestyle = ['-.', '--', ':', '-']
plt.figure(8)
for k in range(0, 4):
    plt.plot(range(0, T_f), P_est[:, k], linestyle=linestyle[k], linewidth=3)
plt.xlabel('Timestep (k)')
plt.ylabel('Uncertainty of Estimated Volume of Water')
plt.title(r'Uncertainty of Estimated Volume of Water '
          'vs Time (Part c)')
plt.legend(labels=['Tank 1', 'Tank 2', 'Tank 3', 'Tank 4'], loc="upper right")

linestyle = ['-', '--', ':', '-']
plt.figure(9)
for k in range(4, 8):
    plt.plot(range(0, T_f), P_est[:, k], linestyle=linestyle[k-4], linewidth=3)
plt.xlabel('Timestep (k)')
plt.ylabel('Uncertainty of Estimated Consumption of Water')
plt.title(r'Uncertainty of Estimated Consumption of Water '
          'vs Time (Part c)')
plt.legend(labels=['Tank 1', 'Tank 2', 'Tank 3', 'Tank 4'], loc="upper right")

print(
    'The estimate uncertainty for tank 1 is lower than the solution we had'
    ' for the simpler network because the system dynamics bewteen tanks are'
    ' coupled. Therefore the additional measurements of added states reduce '
    'the estimate uncertainty of tank 1.')

'''
(c-iii) We will now repeat the previous problem, except that now the sensor
of tank 3 has failed, and thus no longer provides any measurements. Modify
your Kalman filter from before to adjust for the lost sensor, otherwise
using previous data.
'''

# N, d, alpha, and A are same as before, H is redefined below
H = np.concatenate((H[0:2], H[3:4]), axis=0)

# Process noise is the same as before, measurement noise redefined below
W = 25 * np.eye(3)  # z(k) is of size 3

# Process and measurement noise are zero mean, white, and independant

# Initialize estimate and covariance of state (at k = 0)
xm = np.array([[20, 40, 60, 20, 7, 7, 7, 7]]).T
Pm = np.diag([20, 20, 20, 20, 1, 1, 1, 1])

# preallocate parameters
x_est = np.zeros((T_f, N))
x_est[0] = xm.ravel()
P_est = np.zeros((T_f, N))
P_est[0] = Pm.diagonal()

# Given measurements (tank 3 sensor removed)
z = np.array([[62.6, 70.3, 73.5, 77.2, 73.2, 94.2, 87.4, 89.7, 90.4, 94.2],
              [29.4, 44.8, 37.3, 40.1, 44.1, 43.8, 53.8, 49.9, 51.9, 52.1],
              [40.9, 21.9, 18.0, 08.8, 23.9, 17.2, 18.6, 22.0, 22.9, 17.2]])

for k in range(1, T_f):  # Note that estimate for k=0 is initialized above

    # Kalman filter estimation: time update
    xp, Pp = time_update_c(xm, Pm)

    # Kalman filter estimation: measurement update
    # (ensuring z(k) is 3x1 after slicing)
    xm, Pm = meas_update_c(xp, Pp, z[0:3, k-1].reshape(3, 1))

    # Store results for plotting
    x_est[k] = xm.ravel()
    P_est[k] = Pm.diagonal()

#  Plotting
plt.figure(10)
plt.plot(range(0, T_f), x_est[:, 0:4], linewidth=3)
plt.xlabel('Timestep (k)')
plt.ylabel('Estimated Volume of Water')
plt.title(r'Estimated Volume of Water vs Time (Part c-iii)')
plt.legend(labels=['Tank 1', 'Tank 2', 'Tank 3', 'Tank 4'], loc="upper center")

plt.figure(11)
plt.plot(range(0, T_f), x_est[:, 4:9], linewidth=3)
plt.xlabel('Timestep (k)')
plt.ylabel('Estimated Consumption of Water')
plt.title(r'Estimated Consumption of Water vs Time (Part c-iii)')
plt.legend(labels=['Tank 1', 'Tank 2', 'Tank 3', 'Tank 4'], loc="lower left")

linestyle = ['-', '--', '-', '-']
plt.figure(12)
for k in range(0, 4):
    plt.plot(range(0, T_f), P_est[:, k], linestyle=linestyle[k], linewidth=3)
plt.xlabel('Timestep (k)')
plt.ylabel('Uncertainty of Estimated Volume of Water')
plt.title(r'Uncertainty of Estimated Volume of Water '
          'vs Time (Part c-iii)')
plt.legend(labels=['Tank 1', 'Tank 2', 'Tank 3', 'Tank 4'], loc="upper right")

linestyle = ['-', '--', '-', '-']
plt.figure(13)
for k in range(4, 8):
    plt.plot(range(0, T_f), P_est[:, k], linestyle=linestyle[k-4], linewidth=3)
plt.xlabel('Timestep (k)')
plt.ylabel('Uncertainty of Estimated Consumption of Water')
plt.title(r'Uncertainty of Estimated Consumption of Water '
          'vs Time (Part c-iii)')
plt.legend(labels=['Tank 1', 'Tank 2', 'Tank 3', 'Tank 4'], loc="right")

print('losing this sensor has slightly affected our ability to estimate'
      ' the tank levels. As seen by comparing the graphs bewteen (c) and'
      ' (c-iii), Losing the tank 3 sensor moderately increased the uncertainty'
      ' for the estimated consumption and volume of tank 3 only. Uncertainties'
      ' for the remaining 6 states remained relatively unchanged. Estimation'
      ' graphs were affected by removing the sensor, but their overall shapes'
      ' remained consistent. ')

plt.show()

print(0)
