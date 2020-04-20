'''
Created on Apr 15, 2020

@author: Shawn Marshall-Spitzbart

UC Berkeley ME231B Assignment #9 Problem 2
'''
import numpy as np
from scipy.linalg import solve_discrete_are as DARE
from numpy.linalg import inv as inv
import matplotlib.pyplot as plt

'''
Consider a simple double-integrator model for an autonomous electric vehicle.
The position of the vehicle (relative to its target point) along one spatial
axis is given by x1(k), and the velocity in that direction is x2(k).
The system input is the acceleration u(k), as produced by electric motors,
so that the system dynamics are
[[x1(k+1)], [x2(k+1)]] = [[x1(k) + x2(k)*dt + 1/2*u(k)*dt**2], [x2(k) + u(k)*dt]]
where dt = 0.1 is the system sampling time.

You will design an infnite horizon LQR controller for this system,
and compare (via simulation and analysis) the response of the system under
different cost functions. Consider the three different cost functions below,
which differ only in the values used for the quadratic cost matrices Q and R:

(A) R = 1, and Q = diag (2, 0)
(B) R = 1, and Q = diag (4, 0)
(C) R = 2, and Q = diag (4, 0)
(D) R = 1, and Q = diag (2, 2)
(E) R = 1, and Q = [[2, 1], 1, 2]]
'''
configs = ['A', 'B', 'C', 'D', 'E']
dt = 0.1
R = np.array([1, 1, 2, 1, 1])
Q = np.array([np.diag([2, 0]), np.diag([4, 0]), np.diag([4, 0]),
              np.diag([2, 2]), np.array([[2, 1], [1, 2]])])


'''
a. Compute the infnite horizon LQR gain for each cost function. Comment on how
the gains compare for systems A, B, and C. Clearly specify all matrices you use.
'''

# Derive A and B matricies from given dynamics
A, B = np.array([[1, dt], [0, 1]]), np.array([[1/2*dt**2], [dt]])

# Implement Discrete Algrbraic Riccati equation for each case
U_inf, F_inf = np.zeros((len(A), len(A))), np.zeros((len(configs), 1, len(B)))

for i, config in enumerate(configs):
    U_inf = np.array(DARE(A, B, Q[i], R[i]))
    F_inf[i] = np.array(inv(R[i] + B.T @ U_inf @ B) @ B.T @ U_inf @ A)
    print('The infinite horizon LQR gain for part ' + config + ' is '
          + repr([round(comp, 4) for comp in F_inf[i, 0, :]]))

print('\n', 'Configurations (A) and (C) produced the same LQR gains since Q and R'
      ' from config (A) are proportionally scaled by 1/2 compared to Q and R'
      ' from config (C). The LQR gains from config (B) were larger than those'
      ' of (A) and (C), since the first term in the Q matrix is 4 times the R'
      ' term for config (B) and only 2 times the R term in (A) and (C).'
      ' Essentially, it is the scaling between Q and R that'
      ' affects your LQR feedback policy, not the numerical values themselves.'
      ' Configurations (D) and (E) had the same LQR feedback policy,'
      ' suggesting that the non-diagonal terms in Q did not affect the LQR'
      ' solution for this particular system.', '\n')

'''
b. Compute the closed-loop eigenvalues for each system.
'''

# Closed loop LQR system: x[k+1] =  A*x(k) + B*u(k) = (A - B*F_inf)*x(k)

for i, config in enumerate(configs):
    eigs = np.linalg.eig(A - B @ F_inf[i])[0]
    print('The closed loop eigenvalues for part ' + config
          + ' are ' + repr(round(eigs[0], 4)) + ' and ' + repr(round(eigs[1], 4)))

'''
c. Make a graph, with three subplots (position state, velocity state, and
acceleration input). Plot on the same graph the closed-loop system response
from the initial condition x(0) = [[10], [0]] for k = 0 through 20.
'''
k_len, x_len = 20+1, 2


# Initialize state and input arrays
x = np.zeros((len(configs), k_len, x_len, 1))
u_opt = np.zeros((len(configs), k_len, 1, 1))

for i in range(len(configs)):
    # Set IC for each config
    x[i, 0] = np.array([[10], [0]])
    for k in range(k_len-1):
        # Compute optimal input and apply to system at each timestep
        u_opt[i, k] = - F_inf[i] @ x[i, k]
        x[i, k+1] = A @ x[i, k] + B @ u_opt[i, k]


fig, ax = plt.subplots(3, 1, sharex=True)
ax[0].set_title('Closed Loop LQR System Response')

linestyle, colors = ['-', '-', '-.', '-', '-.'], ['y', 'g', 'b', 'c', 'r']
for i, config in enumerate(configs):
    ax[0].plot(range(k_len), x[i, :, 0], label='Part '+config,
               linestyle=linestyle[i], color=colors[i], linewidth=3)
    ax[1].plot(range(k_len), x[i, :, 1], label='true',
               linestyle=linestyle[i], color=colors[i], linewidth=3)
    ax[2].plot(range(k_len), u_opt[i, :, 0, 0], label='true',
               linestyle=linestyle[i], color=colors[i], linewidth=3)

ax[0].legend(loc='best')

ax[-1].set_xlabel('Timestep k')
ax[0].set_ylabel('Position')
ax[1].set_ylabel('Velocity')
ax[2].set_ylabel('Acceleration Input')

plt.show()

