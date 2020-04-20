'''
Created on Apr 19, 2020

@author: Shawn Marshall-Spitzbart
'''
import numpy as np
'''
We have the following discrete-time dynamic system with state x(k) and
measurement z(k). The parameters are A = 0.5, B = 0.4, and H = 1.
The process noise v(k) ~ N(0, 0.2), the measurement noise w(k) ~ N(0, 0.3),
and the initial state x(0) ~ N (1, 0.1). The sequences v(.), w(.), are
independent, and independent of the initial state.

x(k+1) = A*x(k) + B*u(k) + v(k)
z(k) = H*x(k) + w(k)

We are given the following cost function, with S = Q = 1 and R = 2,
which we wish to minimize over the control input u.

J = E[ x(3).T*S*x(3) + sum(x.T*Q*x + u.T*R*u) ]

We are given the measurement z(1) = 0.9, and that the initial input was
u(0) = 0. Compute the optimal control u(1) with respect to the cost J.
'''
A, B, H = 0.5, 0.4, 1.0
V, W = 0.2, 0.3
S, Q, R = 1, 1, 2

'''
Note that this is a stochastic system with imperfect state information.
Linear Quadratic Gaussian Control (LQG) applies. The optimal control input
at time k = 1 is proportional to the estimated state at k = 1.
We first apply a Kalman filter to find this state estimate.
'''


# Define measurement and time update for Kalman filter implementation
def time_update(xm, Pm, u):
    xp = A*xm + B*u
    Pp = A*Pm*A + V
    return xp, Pp


def meas_update(xp, Pp, z):
    K = Pp*H*(H*Pp*H + W)**-1
    xm = xp + K*(z - H*xp)
    Pm = (1 - K*H)*Pp*(1 - K*H) + K*W*K
    return xm, Pm


# Initialize KF estimator with E[x(0)] and Var[x(0)]
E_x0, Var_x0 = 1, 0.1

# Compute x_est for k = 1, given that u(0) = 0 and z(1) = 0.9
xp_1, Pp_1 = time_update(E_x0, Var_x0, 0)
x_est_1, P_est_1 = meas_update(xp_1, Pp_1, 0.9)
print('The estimate for x at k = 1 is ' + repr(round(x_est_1, 4)))

'''
Note that u_opt(k=1) is exclusively a function of E[x(1)],
which is affected by the applied input, u(k=0)
'''

# Initialize Ricatti equation with U(2) = S, iterate backwards for U(1)
U = np.zeros(4)
U[3] = S

for k in range(3, 1, -1):
    U[k-1] = Q + A*U[k]*A - A*U[k]*B*(R + B*U[k]*B)**-1*B*U[k]*A

# Compute feedback gain F(k=1)
F = np.zeros(3)

for k in range(3, 1, -1):
    F[k-1] = (R + B*U[k]*B)**-1*B*U[k]*A

# Implement linear feedback policy to obtain optimal input at k = 1
u_opt_1 = -F[1]*x_est_1

print('The optimal control input, u(1) with respect to the cost J is '
      + repr(round(u_opt_1, 4)))
