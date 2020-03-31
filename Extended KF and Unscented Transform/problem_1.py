'''
Created on Mar 28, 2020

@author: Shawn Marshall-Spitzbart

UC Berkeley ME231B Assignment #7 Problem 1
'''

import numpy as np
import sympy as sp

'''
Design an Extended Kalman Filter (EKF) for the given 2-state nonlinear system
and compute its posteriori estimate xm(1), given z(1) = 0.5.
Usual independance assumptions apply
'''

'''
Define given nonlinear system and measurement model. Note that process and
measureent noise will be set to zero for EKF but is included in function to
abide by math notation. Assume no inputs to system.
'''


def nlsys(xk, vk):  # 'q'
    return np.array([[np.sin(xk[0][0]) + np.cos(xk[1][0]) + vk[0]],
                     [np.cos(xk[0][0]) - np.sin(xk[1][0]) + vk[1]]])


def nlmeas(xk, w):  # 'h
    return xk[0][0] * xk[1][0] + w


# Define global vars
V, W = np.array([[0.3, 0], [0, 0.3]]), 0.2


# (a) Initialization
xm, Pm = np.array([[0.5], [0.5]]), np.array([[0.5, 0], [0, 0.5]])

# (b) Prior update: Jacobian matrices A and L
x1, x2, v1, v2 = sp.symbols('x1 x2 v1 v2')
xk, vk = sp.Matrix([[x1], [x2]]), sp.Matrix([[v1], [v2]])

q = sp.Matrix([[sp.sin(x1) + sp.cos(x2) + v1],
               [sp.cos(x1) - sp.sin(x2) + v2]])

# Take jacobians of nl system model and lambdify to functions of np arrays
A = sp.lambdify([x1, x2, v1, v2], q.jacobian(xk))
L = sp.lambdify([x1, x2, v1, v2], q.jacobian(vk))

# Input values
Ak = A(xm[0][0], xm[1][0], 0, 0)
Lk = L(xm[0][0], xm[1][0], 0, 0)

# Apply linear KF prediction equations
xp, Pp = nlsys(xm, [0, 0]), Ak @ Pm @ Ak.T + Lk @ V @ Lk.T

# (c) Measurement update: Jacobian matrices H and M
w = sp.symbols('w')
h = sp.Matrix([[x1*x2 + w]])

# Take jacobians of nl meas model and lambdify to functions of np arrays
H = sp.lambdify([x1, x2, w], h.jacobian(xk))
M = sp.lambdify([x1, x2, w], h.diff(w))

# Input values
Hk = H(xp[0][0], xp[1][0], 0)
Mk = M(xp[0][0], xp[1][0], 0)

# Apply linear KF measurement update equations
z1 = 0.5
Kk = Pp @ Hk.T @ np.linalg.inv(Hk @ Pp @ Hk.T + Mk * W * Mk)
xm = xp + Kk * (z1 - nlmeas(xp, 0))
Pm = (np.eye(2) - Kk @ Hk) @ Pp

# (d) Highlight final answer xm(k = 1)
print('Posterior estimate, xm(k=1), given z(k=1) is: ' + repr(xm))
