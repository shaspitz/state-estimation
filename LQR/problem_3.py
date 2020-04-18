'''
Created on Apr 16, 2020

@author: Shawn Marshall-Spitzbart

UC Berkeley ME231B Assignment #9 Problem 3
'''
import numpy as np
from numpy.linalg import inv as inv
import matplotlib.pyplot as plt

'''
You are working on the mission design for a probe's deep-space asteroid
rendezvous mission. The mission consists of three stages: a first stage
launches the probe out of Earth's gravity well; a second stage performs a
course interception maneuver; and a third stage does a final fine landing
maneuver. This problem focuses on the second stage, and the goal is to
generate specifications for the first stage (how accurate must the
first stage be so that the second and third can complete the mission?).
The position of the probe along one spatial axis is given by x1(k), and
the velocity in that direction is x2(k). The system input is the acceleration
u(k), as produced by thrusters on the satellite, so that the system dynamics
are:

[[x1(k+1)], [x2(k+1)]] = [[x1(k) + x2(k)*dt + 1/2*u(k)*dt**2], [x2(k) + u(k)*dt]]
where dt = 1 is the system sampling time.

At time k = 0, the probe begins the second stage (i.e. the course interception
maneuver) with state x(0), and our objective is to hand over the probe at time
k = 100, suffciently near the target state so that the final stage can perform
a landing. The target state is at a position and velocity of zero. The fuel use
of the probe, per time-step, is 1/2*u(k)**2. After k = 100, the
fine-maneuvering system takes over, and performs final control of the system,
using remaining fuel in the system, to bring the probe to the asteroid.

The cost for this fine maneuvering is given by 1/2*x(100).T*S*x(100), with
S = [[25, 14], [14, 17]]
so that the total fuel cost is given by
J = 1/2*x(100).T*S*x(100) + 1/2*sum(u(k)**2)
'''

'''
a. Make a 2D graph, with axes corresponding to the states x1(0) and x2(0),
and clearly show the region from which the target can be reached, given a fuel
budget of J = 1. Explain the derivation.
'''

'''
Since the system is linear in state and the cost function is quadratic, LQR
applies. Note that only final state and input are weighted in the cost
function (Q = 0 and R = 1)

This problem is well suited for finite horizon LQR, since the final state
of the system is valued highly (in fact it is the only state in time
weighted in the cost function). Moreover, computing the infinite horizon
LQR gain is not possible when Q = 0.
'''
dt, N, S, Q, R = 1, 100, np.array([[25, 14], [14, 17]]), np.zeros((2, 2)), 1

# Derive A and B matricies from given dynamics
A, B = np.array([[1, dt], [0, 1]]), np.array([[1/2*dt**2], [dt]])


# Finite horizon LQR function
def finite_horizon_LQR(x0, N, Q, R, A, B, part_b):

    # Initialize Ricatti equation with U(N) = S, iterate backwards for U(N-1)
    U = np.zeros((N+1, len(x0), len(x0)))
    U[N] = S

    for k in range(N, 0, -1):
        U[k-1] = Q + A.T @ U[k] @ A - A.T @ U[k] @ B @ inv(R + B.T @ U[k] @ B) @ B.T @ U[k] @ A

    # Compute feedback gain F(k)
    F = np.zeros((N, 1, len(x0)))

    for k in range(N, 0, -1):
        F[k-1] = inv(R + B.T @ U[k] @ B) @ B.T @ U[k] @ A

    # Implement controller (linear feedback gain)
    u_opt, J, x = np.zeros(N), np.zeros(N+1), np.zeros((N+1, len(x0), 1))

    x[0] = x0
    for k in range(N):
        u_opt[k] = -F[k] @ x[k]

        # Compute state and stage cost for each timestep
        x[k+1] = A @ x[k] + B * u_opt[k]
        J[k] = 1/2*u_opt[k]**2

    # Terminal cost
    J[N] = 1/2*x[N].T @ S @ x[N]

    if part_b:
        # Return everything needed for plotting (x1, x2, u, fuel remaining)
        return x[:, 0, :], x[:, 1, :], u_opt, 1 - J.cumsum()

    else:
        # Return total cost to get to final state
        return x[N], J.sum()



###################################################
# Problem here, LQR solution might be wrong? check w/ someone else



x0 = np.array([[30], [0]])
print('x[N] and Total cost: ', finite_horizon_LQR(x0, N, Q, R, A, B, False))

# Discretize and compute initial states from which target can be reached
x1, x2 = np.linspace(0, 1000, 100), np.linspace(0, 1000, 100)
'''
plt.figure(0)
enough_fuel = np.zeros((len(x1), len(x2)), dtype=bool)
# Brute force
for x1_i, x1_val in enumerate(x1):
    for x2_i, x2_val in enumerate(x2):
        if finite_LQR(np.array([[x1_val], [x2_val]]), N, Q, R, A, B)[1] <= 1:
            enough_fuel[x1_i][x2_i] = True
            plt.plot(x1_val, x2_val, 'b')

print(enough_fuel)
plt.show()
'''


'''
b. Consider the initial condition x(0) = [[408], [0]]. First, confirm that this
is just inside the region you computed above. Then, make a plot with four
sub-axes, showing respectively the states x1(k), x2(k), the input u(k), and
the remaining fuel for k = 0 to 99 for a vehicle starting at this
configuration.
'''

x0 = np.array([[408], [0]])
x1, x2, u_opt, fuel = finite_horizon_LQR(x0, N, Q, R, A, B, True)

fig, ax = plt.subplots(4, 1, sharex=True)
ax[0].set_title('Closed Loop LQR System Response')

ax[0].plot(range(N+1), x1, linewidth=3)
ax[1].plot(range(N+1), x2, linewidth=3)
ax[2].plot(range(N), u_opt, linewidth=3)
ax[3].plot(range(N+1), fuel, linewidth=3)

ax[-1].set_xlabel('Timestep k')
ax[0].set_ylabel('Position, x1(k)')
ax[1].set_ylabel('Velocity, x2(k)')
ax[2].set_ylabel('Input, u(k)')
ax[3].set_ylabel('Remaining fuel')

plt.show()



