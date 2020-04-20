'''
Created on Apr 15, 2020

@author: Shawn Marshall-Spitzbart

UC Berkeley ME231B Assignment #9 Problem 1
'''
import numpy as np

'''
Consider the following system:

x(k+1) = 2x(k) + u(k) with associated cost function: J = 1/2*S*x(N)**2
+ 1/2*sum( Q*x(k)**2 + R*u(k)**2 )

where N = 2, S = 2, Q = 1, R = 2.
You are given that x(0) = 2. Compute the optimal inputs u(0), u(1), and
the resulting state x(2).
'''

# System is linear (time invariant) and cost function is quadratic, LQR applies
N, S, Q, R = 2, 2, 1, 2
A, B = 2, 1

# Initialize Ricatti equation with U(2) = S, iterate backwards for U(1)
U = np.zeros(3)
U[N] = S

for k in range(N, 0, -1):
    U[k-1] = Q + A*U[k]*A - A*U[k]*B*(R + B*U[k]*B)**-1*B*U[k]*A

# Compute feedback gain F(k)
F = np.zeros(2)

for k in range(N, 0, -1):
    F[k-1] = (R + B*U[k]*B)**-1*B*U[k]*A

# Implement controller (linear feedback gain)
u_opt, x = np.zeros(2), np.array([2.0, 0, 0])

for k in range(N):
    u_opt[k] = -F[k]*x[k]
    print('Optimal inpunt u(' + repr(k) + ') = ' + repr(round(u_opt[k], 4)))

    # Compute state
    x[k+1] = 2*x[k] + u_opt[k]

print('Resulting state, x(2) = ' + repr(round(x[2], 4)))


