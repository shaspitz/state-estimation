'''
Created on Mar 26, 2020

@author: Shawn Marshall-Spitzbart

UC Berkeley ME231B Assignment #6 Problem 3
'''

import numpy as np
import scipy.linalg as sp

'''
You are investigating various trade-offs in the design of an autonomous blimp.
The system dynamics are given, with the system state x(k)
comprising the height (x1 in units of m) and vertical velocity (x2 in units of
m/s) of the blimp, driven by a random acceleration.
'''

'''
(a) Write a program that computes Jp as a function of the usual Kalman filter
matrices A, V , H, and W. Use this program to confirm that the original system
 has Jp ~= 3:085m
'''

# Define global vars

dt, stdev, m1 = 1/10, 5, 10

# Time-invariant scalar linear system as given in problem statement
N = 2
A = np.array([[1, dt], [0, 1]])
H = np.array([[1, 0]])


def noise_orig_sys():
    # Process and measurement noise covariances
    V = stdev**2*np.array([[1/4*dt**4, 1/2*dt**3], [1/2*dt**3, dt**2]])
    W = m1**2
    return V, W
    # Noises are zero mean, gaussian, and independant


def compute_jp():

    '''
    The limit of Var[x1(k)|z(1:k)] as k->inf will be the upper-left component
    of the steady-state KF posterior variance.
    '''
    if len(H) == 1:  # Check if H is scalar
        Pinf = sp.solve_discrete_are(A.T, H.T, V, W)
        Kinf = Pinf * H * (H * Pinf * H + W) ** -1
    else:
        Pinf = sp.solve_discrete_are(A.T, H.T, V, W)
        Kinf = Pinf @ H @ np.linalg.inv(H @ Pinf @ H + W)
    '''
    Use time update to transform Pinf (Pp as k -> inf)
    into KF posterior variance (Pm as k -> inf)
    '''
    Pm_inf = (np.eye(N) - Kinf*H)*Pinf*(np.eye(N) - Kinf*H) + Kinf*W*Kinf

    # Extract upper-left term in Pm_inf to obtain Jp
    return np.sqrt(Pm_inf[0, 0])


V, W = noise_orig_sys()
print('The original system has Jp ~= ' + repr(round(compute_jp(), 3)) + 'm')

'''
We are considering various ways to improve the system's estimation performance,
enumerated below. Each modification is an independent modification of the
original system. For each of the following suggested modifications to the
system, compute the performance metric Jp
'''

'''
(b) Replace the GPS sensor with that of brand A, which has m1 = 5m
(the noise standard deviation is cut in half).
'''
m1 = 5
V, W = noise_orig_sys()
print('For part (b) Jp ~= ' + repr(round(compute_jp(), 3)) + 'm')

'''
(c) Replace the GPS sensor with that of brand B, which has dt = 1/20 sec,
and m1 = 10m (i.e. the sensor returns equally noisy measurements,
 but twice as frequently).
'''
dt, m1 = 1/20, 10
A = np.array([[1, dt], [0, 1]])
V, W = noise_orig_sys()
print('For part (c) Jp ~= ' + repr(round(compute_jp(), 3)) + 'm')

'''
(d) Retain the original GPS sensor, and add an airspeed sensor which gives the
additional measurement z2(k) = x2(k) + w2(k), where w2(k) is a zero-mean,
white noise sequence, independent of all quantities, and with
Var [w2(k)] = m2 ** 2 with m2 = 1m/s. The sensors z1 and z2 return data
at the same instants of time.
'''
dt, m1, m2, H = 1/10, 10, 1, np.eye(N)
A = np.array([[1, dt], [0, 1]])


def noise_part_d():
    # Process and measurement noise covariances
    V = stdev**2*np.array([[1/4*dt**4, 1/2*dt**3], [1/2*dt**3, dt**2]])
    W = np.array([[m1**2, 0], [0, m2**2]])
    return V, W
    # Noises are zero mean, gaussian, and independant


V, W = noise_part_d()
print('For part (d) Jp ~= ' + repr(round(compute_jp(), 3)) + 'm')

'''
(e) Retain the original GPS sensor, and add a second, independent barometric
height sensor z2(k) = x1(k) + w2(k), where w2(k) is a zero-mean, white noise
sequence, independent of all quantities, and with Var [w2(k)] = m2 ** 2
with m2 = 10m. The sensors z1 and z2 return data at the same instants of
time.
'''

m2, H = 10, np.array([[1, 0], [1, 0]])
V, W = noise_part_d()
print('For part (e) Jp ~= ' + repr(round(compute_jp(), 3)) + 'm')

'''
(f) Retain the original GPS sensor, and add a second identical GPS sensor from
the same manufacturer, so that z2(k) = x1(k) + w2(k), where w2(k) = w1(k).

Note: This problem can be interpreted two different ways. If the statement
"w2(k) = w1(k)" means that these RVs have the same distributions but are
independant, then the answer to part (f) is as follows.
'''


def noise_part_f_1():
    # Process and measurement noise covariances
    V = stdev**2*np.array([[1/4*dt**4, 1/2*dt**3], [1/2*dt**3, dt**2]])
    W = m1**2*np.eye(N)
    return V, W


V, W = noise_part_f_1()
print('For part (f) interpretation 1, Jp ~= ' + repr(round(compute_jp(), 3))
      + 'm')

'''
If instead the statement "w2(k) = w1(k)" means that these two RV's values are
always equivalent, then the answer to part (f) is a little more complicated.
When w2(k) = w1(k) = m1**2, the covariance matrix for W should be,
m1**2 * [[1, 1], [1, 1]]
However, this covariance matrix is not solvable by the DARE. If we instead
perturb the diagonal terms slightly, then the DARE will solve, but will
approach infinity as the diagonal terms approach 1.
'''


def noise_part_f_2(delta):
    # Process and measurement noise covariances
    V = stdev**2*np.array([[1/4*dt**4, 1/2*dt**3], [1/2*dt**3, dt**2]])
    W = m1**2*np.array([[1 + delta, 1], [1, 1 + delta]])
    return V, W


delta = 0.1
for i in range(0, 5):
    V, W = noise_part_f_2(delta)
    print('When delta = ' + repr(delta)
          + ', Jp ~= ' + repr(round(compute_jp(), 3))
          + 'm')
    delta /= 10

'''
Therefore, for part (f) it makes the most sense to say that Jp -> infinity
when w2(k) = w1(k). This makes sense because we have always assumed that
random noise variables are independant in order
for KF to have a solution. In this case, w1(k) and w2(k) are surely not
independant.
'''
print('For part (f) interpretation 2, Pinf does not converge, Jp -> infinity')

'''
(g) Retain the original sensor, but modify the system design by making
it more aerodynamic and thereby reducing the effect of aerodynamic
disturbances, so that stdev = 1m/s.
'''

stdev, H = 1, np.array([[1, 0]])
V, W = noise_orig_sys()
print('For part (g) Jp ~= ' + repr(round(compute_jp(), 3)) + 'm')
