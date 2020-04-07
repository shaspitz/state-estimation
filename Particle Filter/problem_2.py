'''
Created on Apr 3, 2020

@author: Shawn Marshall-Spitzbart

UC Berkeley ME231B Assignment #8 Problem 2
'''

import numpy as np
import matplotlib.pyplot as plt

'''
In this problem we investigate the randomness of the particle filter.
Consider the following system:
x(k) = x(k-1)^3 + v(k-1)
z(k) = x(k)^3 + w(k)
where x(0), v(.), and w(.) are independent, and each is drawn from
a uniform distribution in the range [-1; 1]. At k = 1, you make the
observation z(1) = 0.5. Implement a particle filter with Np particles,
and use this to approximate the MMSE estimator for the distribution
f(x(1)|z(1)) (by taking the average of the particles after the
resampling step). For each value of Np,
run the particle filter 10^3 times, and record the final estimate for
each run (this should be different, because each run is generated using
different random particles).
'''


'''
f(z(1)=0.5|xp) is computed analytically and its solution is implemented below.
Using the measurement model, z(k) = h(x,w), x(k) becomes a parameter, then I
used change of variables over w(k) through the nonlinear funciton h.
The solution for f(z|x) is applied to all particles in the measurment update
section of the PF algorithm.
'''


def meas_likelihood(xp):
    if -1 <= z1 - xp and z1 - xp <= 1:
        return 1 / (b - a)
    else:
        return 0


# Particle settings, measurement, and uniform distribution bounds
Np, runs = [10, 10**2, 10**3], 10**3
z1, a, b = 0.5, -1, 1

# initialize array for final estimate, x_pf at k = 1
x_est_pf = np.zeros((len(Np), runs))

for Np_iter, N in enumerate(Np):
    for run in range(runs):

        # Initialization
        xm = np.random.uniform(a, b, N)

        # Prior update/prediction step
        vk = np.random.uniform(a, b, N)
        xp = xm**3 + vk

        # Scale particles by measurement likelihood and apply normalization constant
        beta = np.array([meas_likelihood(xp_n) for xp_n in xp])
        alpha = np.sum(beta)
        beta /= alpha

        # Resampling
        beta_sum = np.cumsum(beta)
        xm = np.zeros(N)
        for particle_iter in range(N):
            r = np.random.uniform()
            # first occurance where beta_sum[n_index] >= r
            n_index = np.nonzero(beta_sum > r)[0][0]
            xm[particle_iter] = xp[n_index]

        # Store approximation of MMSE estimate for each run
        x_est_pf[Np_iter][run] = np.mean(xm)

'''
(a) Make a single graph, with three histograms on it overlaid,
showing the distribution of the final error.
'''
num_bins = 10
plt.figure(0)

for i, hist_color in enumerate(['blue', 'red', 'green']):
    plt.hist(x_est_pf[i], num_bins, facecolor=hist_color, edgecolor='black',
             alpha=0.2 + 0.2*i)

plt.xlabel('Paricle Filter Estimate at k = 1')
plt.ylabel('Count')
plt.title(r'Distribution of Final Estimate for PF'
          + ' with ' + repr(num_bins) + ' bins', fontsize=10)
plt.legend(labels=['Np = ' + repr(Np[0]), 'Np = ' + repr(Np[1]),
                   'Np = ' + repr(Np[2])], loc="best")

'''
(b) What is the mean and standard deviation of the particle filter estimates
for each different choice of Np? Comment on this. As an engineer, what is
the effect of increasing Np?.
'''
for Np_iter, N in enumerate(Np):
    print('For Np = ' + repr(N) + ' the mean of PF estimates is '
          + repr(round(np.mean(x_est_pf[Np_iter, :]), 4))
          + ' and the standard deviation of PF estimates is '
          + repr(round(np.std(x_est_pf[Np_iter, :]), 4)))

print('For each value of Np, the mean of PF estimates was approximately'
      ' centered around zero, the standard deviation of estimates'
      ' decreased as Np increased.'
      ' From these results we can infer that as you increase the number of'
      ' particles in your particle filter, your estimate will be more certain.'
      ' This makes sense intuitively because all CRVs are drawn from a uniform'
      ' distribution in the range [-1, 1], therefore estimate distributions'
      ' should be centered around zero, with more estimates lying near zero'
      ' as the number of particles is increased.')

plt.show()
