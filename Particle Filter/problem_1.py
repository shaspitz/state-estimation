'''
Created on Apr 3, 2020

@author: Shawn Marshall-Spitzbart

UC Berkeley ME231B Assignment #8 Problem 1
'''

import numpy as np
import matplotlib.pyplot as plt

'''
In this problem we compare the approximate result of the particle filter
to a Bayesian problem we studied in an earlier homework. Specifcally,
 we revisit Problem 1 of Homework 5: Consider the following system:
x(k) = x(k-1) + v(k-1)
z(k) = x(k) + w(k)
where x(0), v(.), w(.) are independent, and each is drawn from a uniform
distribution in the range [-1; 1]. At k = 1, you make the observation
z(1) = 1.Implement a particle with 104 particles, and use this to approximate
the PDF f(x(1)jz(1)). Make a single graph, showing the analytical solution
to the PDF (as you solved in the earlier homework) and a histogram
of the particles. Comment on the agreement/differences between the two.
'''

# Define global vars

# Scalar linear system and measurement model, 10^4 particles with given bounds
A, H = 1, 1
N, a, b = 10**4, -1, 1

# Initialization
xm = np.random.uniform(a, b, N)

# Prior update/prediction step
vk = np.random.uniform(a, b, N)
xp = A*xm + vk

# Measurement update step
z = 1


# f(z(1)=1|xp) = 1/(b-a) if z is within +-1 of xp, otherwise f(z(1)=1|xp) = 0
def meas_likelihood(xp):
    if abs(z - xp) < 1:
        return 1/(b - a)
    else:
        return 0


# Scale particles by measurement likelihood and apply normalization constant
beta = np.array([meas_likelihood(xp_n) for xp_n in xp])
alpha = np.sum(beta)
beta /= alpha

# Resampling
beta_sum = np.cumsum(beta)
xm = np.zeros(N)
for i in range(N):
    r = np.random.uniform()
    # first occurance where beta_sum[n_index] >= r
    n_index = np.nonzero(beta_sum > r)[0][0]
    xm[i] = xp[n_index]

# Plotting
num_bins = 20

# Enforce valid x(1) values
x = np.linspace(-2, 2, N)

# Analytic answer from HW5 problem 1
analytical_sltn = [np.int(i >= 0) for i in x] * (1 - x/2)

plt.figure(0)
plt.hist(xm, num_bins, facecolor='blue', edgecolor='black', alpha=1,
         density=True)
plt.plot(x, analytical_sltn, linewidth=3)
plt.xlabel('x(1)')
plt.ylabel('f(x(1)|z(1))')
plt.title(r'PF Approximation vs Analytical Solution of f(x(1)|z(1))'
          + ' with bin size = ' + repr(2/num_bins), fontsize=10)
plt.legend(labels=['Analytical Solution', 'Particle Filter Approximation'],
           loc="best")

print('The analytical solution to the PDF f(x(1)|z(1)) and the approximation'
      ' given by the particle filter are generally similar in shape. Clearly'
      ' the approximation was not perfect, but if we increased the number of'
      ' particles used in the PF algorithm, it\'s approximation would approach'
      ' the analytical solution given by the optimal estimator')

plt.show()


