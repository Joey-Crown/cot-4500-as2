import numpy as np

# Question 1
neville_x = np.array([3.6, 3.8, 3.9])
neville_f = np.array([1.675, 1.436, 1.318])
neville_w = 3.7

neville_Q = np.zeros([3, 3], dtype=float)
neville_Q[:, 0] = neville_f


def nevilles_method(q, x, w):
    for i in range(1, len(q)):
        for j in range(1, i + 1):
            t1 = (w - x[i - j]) * q[i][j - 1]
            t2 = (w - x[i]) * q[i - 1][j - 1]
            q[i][j] = (t1 - t2) / (x[i] - x[i - j])
    print(q[len(q) - 1][len(q) - 1], end="\n\n")


nevilles_method(neville_Q, neville_x, neville_w)

# Question 2
newton_fwd_x = np.array([7.2, 7.4, 7.5, 7.6])
newton_fwd_F = np.zeros([4, 4], dtype=float)

newton_fwd_F[:, 0] = np.array([23.5492, 25.3913, 26.8224, 27.4589])


def newtons_fwd_diff(f, x):
    for i in range(1, len(f)):
        for j in range(1, i + 1):
            f[i][j] = (f[i][j - 1] - f[i - 1][j - 1]) / (x[i] - x[i - j])


newtons_fwd_diff(newton_fwd_F, newton_fwd_x)
ans_q2 = np.diagonal(newton_fwd_F[1:, 1:])
print('[%.15f, %.14f, %.14f]' % (ans_q2[0], ans_q2[1], ans_q2[2]), end="\n\n")

# Question 3
approx_w = 7.3


def polynomial_approx(f, x, w):
    # initial values
    p = f[0][0]
    dx = 1
    for i in range(1, len(f)):
        dx *= w - x[i - 1]
        t1 = p
        t2 = f[i][i] * dx
        p = t1 + t2
    return p


ans_q3 = polynomial_approx(newton_fwd_F, newton_fwd_x, approx_w)
print(ans_q3, end="\n\n")

#Question 4
hermite_x = np.array([3.6, 3.8, 3.9])
hermite_f = np.array([1.675, 1.436, 1.318])
hermite_dx = np.array([-1.195, -1.188, -1.182])

# create table
hermite_table = np.zeros([6, 6])
hermite_table[:, 0] = np.repeat(hermite_x, 2)
hermite_table[:, 1] = np.repeat(hermite_f, 2)
print(hermite_table)

