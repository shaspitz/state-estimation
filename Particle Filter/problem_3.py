'''
Created on Apr 5, 2020

@author: Shawn Marshall-Spitzbart

UC Berkeley ME231B Assignment #8 Problem 3
'''

import numpy as np
import matplotlib.pyplot as plt
import timeit

'''
In this problem, we will investigate the computational requirements for the
particle filter as presented in class, and the curse of dimensionality.
We will consider systems of chained integrators, where all noises are Gaussian,
so that we may use the Kalman filter to compute the exact solution to the
estimation problem. For each of the below cases, do the following:
Use the sequence of measurements tabulated below, and compute the exact
conditional mean and conditional variance using the Kalman filter. Then,
implement a particle filter as presented in class, and compare how
the computation time and estimate varies as a function of the number
of particles used. As a metric, we will compare the estimate from the particle
filter to that of the Kalman filter in the sense of the Mahalanobis distance.
The particle filter estimate is computed as the sample average over all
particles after using the measurement at time k = 5, and re-sampling.
Note that because the particle filter is a random algorithm, we will run it
100 times for each configuration, and compute the average Malahobis distance
over the multiple trials. In each case, we have normal distributions for noise
and initial state.
'''

# Given measurements, Np values, and number of runs
zk, Np_list, runs = [1.0, 0.5, 1.5, 1.0, 1.5], [1, 10, 100, 1000], 100


# Define functions for Kalman filter implementation
def time_update(xm, Pm, A):
    if state_len == 1:
        xp = A*xm
        Pp = A*Pm*A + V
    else:
        xp = A @ xm
        Pp = A @ Pm @ A.transpose() + V
    return xp, Pp


def meas_update(xp, Pp, z, H):
    if state_len == 1:
        K = Pp*H*(H*Pp*H + W)**-1
        xm = xp + K*(z - H*xp)
        Pm = (1 - K*H)*Pp*(1 - K*H) + K*W*K
    else:
        K = Pp @ H.T @ np.linalg.inv(H @ Pp @ H.T + W)
        xm = xp + K @ (z - H @ xp)
        Pm = (np.eye(state_len) - K @ H) @ Pp @ (np.eye(state_len) - K @ H).T
        + K * W * K.T
    return xm, Pm


def KF(A, H):

    # Initialize estimate and covariance of state (at k = 0)
    if state_len == 1:
        xm, Pm = 0, 1
    else:
        xm, Pm = np.zeros((state_len, 1)), np.eye(state_len)

    for k in range(5):

        # Kalman filter estimation: time update
        xp, Pp = time_update(xm, Pm, A)

        # Kalman filter estimation: measurement update
        xm, Pm = meas_update(xp, Pp, zk[k], H)

    # Return exact conditional mean and conditional variance
    return xm, Pm


# Define function for particle filter implementation
def PF(A, Np, meas_likelihood):

    if state_len == 1:

        t_start = timeit.default_timer()

        for k in range(5):

            # Initialization
            xm = np.random.normal(0, 1, Np)
            
            # Prior update/prediction step
            vk = np.random.normal(0, V, Np)
            xp = A*xm + vk

            # Measurement update step

            # Scale particles by meas likelihood and apply normalization constant
            beta = np.array([meas_likelihood(xp_n, zk[k]) for xp_n in xp])
            alpha = np.sum(beta)
            beta /= alpha

            # Resampling
            beta_sum = np.cumsum(beta)
            xm = np.zeros(Np)
            for i in range(Np):
                r = np.random.uniform()
                # first occurance where beta_sum[n_index] >= r
                n_index = np.nonzero(beta_sum > r)[0][0]
                xm[i] = xp[n_index]

    else:
        t_start = timeit.default_timer()

        for k in range(5):

            # Initialization
            xm = np.random.normal(0, 1, (state_len, 1, Np))

            # Prior update/prediction step
            vk = np.random.normal(0, V[state_len-1][state_len-1], Np)
            vk_vec = np.zeros((state_len, 1, Np))
            vk_vec[state_len-1, 0, :] = vk
            xp = [A @ xm[:, :, N] + vk_vec[:, :, N] for N in range(Np)]

            # Measurement update step

            # Scale particles by meas likelihood and apply normalization constant
            beta = np.array([meas_likelihood(xp_n, zk[k]) for xp_n in xp])
            alpha = np.sum(beta)
            beta /= alpha

            # Resampling
            beta_sum = np.cumsum(beta)
            xm = np.zeros((state_len, 1, Np))
            for i in range(Np):
                r = np.random.uniform()
                # first occurance where beta_sum[n_index] >= r
                n_index = np.nonzero(beta_sum > r)[0][0]
                xm[:, :, i] = xp[n_index]

    # Particle filter estimate, variance, and computation time
    return np.mean(xm), timeit.default_timer() - t_start


# Define Malahobis distance
def dm(xm_kf, Pm_kf, xm_pf):
    if state_len == 1:
        return (xm_kf - xm_pf)**2 * Pm_kf**-1
    else:
        return (xm_kf - xm_pf).T @ np.linalg.inv(Pm_kf) @ (xm_kf - xm_pf)


'''
(a) Consider the scalar system, x(k) = x(k-1) + v(k-1) and z(k) = x(k) + w(k)
'''
print('Part (a):')
state_len, A, H = 1, 1, 1

# Time-invariant process and measurement noise variances (Gaussian)
V, W = 1, 1

# Kalman Filter
xm_kf_a, Pm_kf_a = KF(A, H)


# Particle Filter
# Define measurement likeleyhood, f(z|x), for PF using change of variables
def meas_likelihood_a(xp, zk):
    # return f(w|x) = f(w) = f(z-x)
    return 1/np.sqrt(2*np.pi*W)*np.exp(-1/(2*W)*(zk - xp)**2)


xm_pf_a = np.zeros((len(Np_list), runs))
comp_time_a = np.zeros((len(Np_list), runs))
dm_a = np.zeros(len(Np_list))

# Compute PF estimates and computation times
for Np_iter, N in enumerate(Np_list):
    for run in range(runs):
        xm_pf_a[Np_iter][run], comp_time_a[Np_iter][run] = PF(A, N, meas_likelihood_a)

    # Compute average Malahobis distance and over each set of 100 runs
    dm_a[Np_iter] = np.mean([dm(xm_kf_a, Pm_kf_a, xm_pf) for xm_pf in xm_pf_a[Np_iter]])

for Np_iter, N in enumerate(Np_list):

    print('For Np = ' + repr(N) + ', the average Malahobis distance comparing'
          ' the PF estimate to the KF estimate over 100 simulations was '
          + repr(round(np.mean(dm_a[Np_iter]), 4))
          + ' and the average computation time over 100 simulations was '
          + repr(round(np.mean(comp_time_a[Np_iter]), 6)) + ' seconds.')

'''
(b) Consider a double integrator, so that
x1(k) = x1(k-1) + x2(k-1) and x2(k) = x2(k-1) + v(k-1); z(k) = x1(k) + w(k)
'''
print('Part (b):')
state_len, A, H = 2, np.array([[1, 1], [0, 1]]), np.array([[1, 0]])

# Time-invariant process and measurement noise variances (Gaussian)
V, W = np.array([[0, 0], [0, 1]]), 1

# Kalman Filter
xm_kf_b, Pm_kf_b = KF(A, H)


# Particle Filter
# Define measurement likeleyhood, f(z|x), for PF using change of variables
def meas_likelihood_b(xp, zk):
    # return f(w|x) = f(w) = f(z-x)
    return 1/(2*np.pi*W)*np.exp(-1/(2*W)*(zk - xp[0][0])**2)


xm_pf_b = np.zeros((len(Np_list), runs))
comp_time_b = np.zeros((len(Np_list), runs))
dm_b = np.zeros(len(Np_list))

# Compute PF estimates and computation times
for Np_iter, N in enumerate(Np_list):
    for run in range(runs):
        xm_pf_b[Np_iter][run], comp_time_b[Np_iter][run] = PF(A, N, meas_likelihood_b)

    # Compute average Malahobis distance and over each set of 100 runs
    dm_b[Np_iter] = np.mean([dm(xm_kf_b, Pm_kf_b, xm_pf) for xm_pf in
                             xm_pf_b[Np_iter]])

for Np_iter, N in enumerate(Np_list):

    print('For Np = ' + repr(N) + ', the average Malahobis distance comparing'
          ' the PF estimate to the KF estimate over 100 simulations was '
          + repr(round(np.mean(dm_b[Np_iter]), 4))
          + ' and the average computation time over 100 simulations was '
          + repr(round(np.mean(comp_time_b[Np_iter]), 6)) + ' seconds.')


'''
(c) Consider a quadruple integrator (a reasonable model for a quadcopter's
horizontal dynamics) x(k) = [[x1], [x2], [x3], [x4]]
with x(k) = [[x1(k-1) + x2(k-1)],
            [x2(k-1) + x3(k-1)],
            [x3(k-1) + x4(k-1)],
            [x4(k-1) + v(k-1)]]
'''
print('Part (c):')
state_len, A, H = 4, np.array([[1, 1, 0, 0],
                               [0, 1, 1, 0],
                               [0, 0, 1, 1],
                               [0, 0, 0, 1]]), np.array([[1, 0, 0, 0]])

V, W = np.array([[0, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 1]]), 1

# Kalman Filter
xm_kf_c, Pm_kf_c = KF(A, H)

# Particle Filter
# Measurement likeleyhood, f(z|x) is defined the same as (b)
meas_likelihood_c = meas_likelihood_b

xm_pf_c = np.zeros((len(Np_list), runs))
comp_time_c = np.zeros((len(Np_list), runs))
dm_c = np.zeros(len(Np_list))

# Compute PF estimates and computation times
for Np_iter, N in enumerate(Np_list):
    for run in range(runs):
        xm_pf_c[Np_iter][run], comp_time_c[Np_iter][run] = PF(A, N, meas_likelihood_c)

    # Compute average Malahobis distance and over each set of 100 runs
    dm_c[Np_iter] = np.mean([dm(xm_kf_c, Pm_kf_c, xm_pf) for xm_pf in xm_pf_c[Np_iter]])

for Np_iter, N in enumerate(Np_list):

    print('For Np = ' + repr(N) + ', the average Malahobis distance comparing'
          ' the PF estimate to the KF estimate over 100 simulations was '
          + repr(round(np.mean(dm_c[Np_iter]), 4))
          + ' and the average computation time over 100 simulations was '
          + repr(round(np.mean(comp_time_c[Np_iter]), 6)) + ' seconds.')

# We plot malahobis distance because it shows how far the distributions
# are from one another??

# Plotting
plt.figure(0)
for dm in [dm_a, dm_b, dm_c]:
    plt.plot(Np_list, dm, linewidth=3)
plt.xlabel('Number of Particles, Np')
plt.ylabel('Average PF Error Represented as Malahobis Distance')
plt.title(r'Average PF Error Represented as Malahobis Distance vs Number of'
          ' Particles', fontsize=10)
plt.xscale('log')
plt.yscale('log')
plt.legend(labels=['Part (a)', 'Part (b)', 'Part (c)'], loc="best")

plt.figure(1)
linestyle = ['-', '-', '--']
for i, comp_time in enumerate([comp_time_a, comp_time_b, comp_time_c]):
    plt.plot(Np_list, [np.mean(comp_time[i]) for i in range(len(Np_list))],
             linewidth=3, linestyle=linestyle[i])
plt.xlabel('Number of Particles, Np')
plt.ylabel('Average PF Computation Time (sec)')
plt.title(r'Average PF Computation Time vs Number of Particles', fontsize=10)
plt.xscale('log')
plt.yscale('log')
plt.legend(labels=['Part (a)', 'Part (b)', 'Part (c)'], loc="best")

plt.show()
