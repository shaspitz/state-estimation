'''
Created on Apr 29, 2020

@author: shawn
'''

import numpy as np


# Unscented Transform Function (expectation of state, variance of state,
# state length 'n', nonlinear funciton 'g' which takes state as input)
# INPUT EVERYTHING AS NP ARRAYS (2D column vector for state)
def unscented_transform(E_x, Var_x, g):
    n = len(E_x)

    sx = np.zeros((2*n, n, 1))
    for i in range(n):
        sx[i] = E_x + np.linalg.cholesky(n*Var_x)[:, i].reshape((n, 1))
        sx[n+i] = E_x - np.linalg.cholesky(n*Var_x)[:, i].reshape((n, 1))

    sy = np.zeros((2*n, n, 1))
    for i in range(2*n):
        sy[i] = g(sx[i])

    E_y = np.zeros((n, 1))
    for i in range(2*n):
        E_y += 1/(2*n)*sy[i]

    Var_y = np.zeros((n, n))
    for i in range(2*n):
        Var_y += 1/(2*n)*(sy[i] - E_y)@(sy[i] - E_y).T

    return E_y, Var_y

# Example:
def q_b(x): return (x - 1)**3
E_x_example, Var_x_example = unscented_transform(np.array([[1]]), np.array([[4]]), q_b)
print(E_x_example, Var_x_example)