'''
Created on Mar 31, 2020

@author: Shawn Marshall-Spitzbart

UC Berkeley ME231B Assignment #7 Problem 3
'''
import numpy as np
import sympy as sp

'''
This problem will look at three different systems with deterministic dynamics,
but uncertain initial state. In each case, E[x(0)] = 1 and Var [x(0)] = 4,
and we are interested in x(1) = q(x(0)). We will consider three
different approaches to predict E[x(1)] and Var [x(1)], described below.
'''

E_x, Var_x = 1, 4

print('Part (a):')


# Consider q(x) = -x + 2|x|
def q_a(x): return -x + 2*abs(x)


# (a-i) Use techniques from the EKF (linearization), create function for dq/dx
def A_a(x):
    if x >= 0:
        return 1
    else:
        return -3


# Prediction (with x0 = E[x0], and P0 = Var[x0])
xp, Pp = q_a(E_x), A_a(E_x) * Var_x * A_a(E_x)

print('Using techniques from the EKF, E[x(1)] = ' + repr(round(xp, 4))
      + ' and Var[x(1)] = ' + repr(round(Pp, 4)))

# (a-ii) Use the unscented transform

# The state, x, is scalar so we will have sigma points 0 and (2(1)-1) = 1
sx0, sx1 = E_x + np.sqrt(Var_x), E_x - np.sqrt(Var_x)

# Transform sigma-points through nl function
sy0, sy1 = q_a(sx0), q_a(sx1)

# Compute approximation for resulting mean and variance
xp, Pp = np.mean([sy0, sy1]), np.var([sy0, sy1])

print('Using techniques from the unscented transform, E[x(1)] = '
      + repr(round(xp, 4)) + ' and Var[x(1)] = ' + repr(round(Pp, 4)))

'''
(a-iii) Numerically approximate the statistics by repeated sampling.
Generate 10**6 samples, and do this for both x(0) normally distributed,
and uniformly distributed.
'''
samp = 10**6


# Function to compute uniform distrubution bounds from sample statistics
def get_bounds(mean, var):
    a, b = sp.Symbol("a"), sp.Symbol("b")
    eqtns = (sp.Eq((a + b) / 2, mean), sp.Eq((b - a)**2 / 12, var))
    ans = sp.solve(eqtns, (a, b))
    return ans[0][0], ans[0][1]  # return first answer, 'b' being larger


# Generate samples for x0 being normally and uniformly distributed
a, b = get_bounds(E_x, Var_x)
x0_norm = np.random.normal(E_x, np.sqrt(Var_x), samp)
x0_uni = np.random.uniform(a, b, samp)

# Transform x0 in x1 with nonlinear function and compute resulting statistics
x1_norm, x1_uni = q_a(x0_norm), q_a(x0_uni)
xp_norm, Pp_norm = np.mean(x1_norm), np.var(x1_norm)
xp_uni, Pp_uni = np.mean(x1_uni), np.var(x1_uni)

print('Using repeated sampling with x0 normally distributed, E[x(1)] = '
      + repr(round(xp_norm, 4)) + ' and Var[x(1)] = '
      + repr(round(Pp_norm, 4)))

print('Using repeated sampling with x0 uniformly distributed, E[x(1)] = '
      + repr(round(xp_uni, 4)) + ' and Var[x(1)] = ' + repr(round(Pp_uni, 4)))


print('Part (b):')


# Consider q(x) = (x - 1)**3
def q_b(x): return (x - 1)**3


# (b-i) Use techniques from the EKF (linearization), create function for dq/dx
def A_b(x): return 3*(x-1)**2


# Prediction (with x0 = E[x0], and P0 = Var[x0])
xp, Pp = q_b(E_x), A_b(E_x) * Var_x * A_b(E_x)

print('Using techniques from the EKF, E[x(1)] = ' + repr(round(xp, 4))
      + ' and Var[x(1)] = ' + repr(round(Pp, 4)))

# (b-ii) Use the unscented transform

# The state, x, is still scalar so we can use same sx0 and sx1 points as above

# Transform sigma-points through nl function
sy0, sy1 = q_b(sx0), q_b(sx1)

# Compute approximation for resulting mean and variance
xp, Pp = np.mean([sy0, sy1]), np.var([sy0, sy1])

print('Using techniques from the unscented transform, E[x(1)] = '
      + repr(round(xp, 4)) + ' and Var[x(1)] = ' + repr(round(Pp, 4)))

'''
(b-iii) Numerically approximate the statistics by repeated sampling.
Generate 10**6 samples, and do this for both x(0) normally distributed,
and uniformly distributed.
'''

# Use above generated samples for x0 being normally and uniformly distributed

# Transform x0 in x1 with nonlinear function and compute resulting statistics
x1_norm, x1_uni = q_b(x0_norm), q_b(x0_uni)
xp_norm, Pp_norm = np.mean(x1_norm), np.var(x1_norm)
xp_uni, Pp_uni = np.mean(x1_uni), np.var(x1_uni)

print('Using repeated sampling with x0 normally distributed, E[x(1)] = '
      + repr(round(xp_norm, 4)) + ' and Var[x(1)] = '
      + repr(round(Pp_norm, 4)))

print('Using repeated sampling with x0 uniformly distributed, E[x(1)] = '
      + repr(round(xp_uni, 4)) + ' and Var[x(1)] = ' + repr(round(Pp_uni, 4)))


print('Part (c):')


# Consider q(x) = 3x
def q_c(x): return 3*x


# (c-i) Use techniques from the EKF (linearization), create function for dq/dx
def A_c(): return 3


# Prediction (with x0 = E[x0], and P0 = Var[x0])
xp, Pp = q_c(E_x), A_c() * Var_x * A_c()

print('Using techniques from the EKF, E[x(1)] = ' + repr(round(xp, 4))
      + ' and Var[x(1)] = ' + repr(round(Pp, 4)))

# (c-ii) Use the unscented transform

# The state, x, is still scalar so we can use same sx0 and sx1 points as above

# Transform sigma-points through nl function
sy0, sy1 = q_c(sx0), q_c(sx1)

# Compute approximation for resulting mean and variance
xp, Pp = np.mean([sy0, sy1]), np.var([sy0, sy1])

print('Using techniques from the unscented transform, E[x(1)] = '
      + repr(round(xp, 4)) + ' and Var[x(1)] = ' + repr(round(Pp, 4)))

'''
(c-iii) Numerically approximate the statistics by repeated sampling.
Generate 10**6 samples, and do this for both x(0) normally distributed,
and uniformly distributed.
'''

# Use above generated samples for x0 being normally and uniformly distributed

# Transform x0 in x1 with nonlinear function and compute resulting statistics
x1_norm, x1_uni = q_c(x0_norm), q_c(x0_uni)
xp_norm, Pp_norm = np.mean(x1_norm), np.var(x1_norm)
xp_uni, Pp_uni = np.mean(x1_uni), np.var(x1_uni)

print('Using repeated sampling with x0 normally distributed, E[x(1)] = '
      + repr(round(xp_norm, 4)) + ' and Var[x(1)] = '
      + repr(round(Pp_norm, 4)))

print('Using repeated sampling with x0 uniformly distributed, E[x(1)] = '
      + repr(round(xp_uni, 4)) + ' and Var[x(1)] = ' + repr(round(Pp_uni, 4)))

