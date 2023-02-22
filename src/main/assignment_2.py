from polynomial import *

np.set_printoptions(precision=7, suppress=True, linewidth=100)


# Question 1
neville_x = np.array([3.6, 3.8, 3.9])
neville_f = np.array([1.675, 1.436, 1.318])
neville_w = 3.7

neville_Q = np.zeros([3, 3], dtype=float)
neville_Q[:, 0] = neville_f

ans_q1 = nevilles_method(neville_Q, neville_x, neville_w)
print(ans_q1, end="\n\n")


# Question 2
newton_fwd_x = np.array([7.2, 7.4, 7.5, 7.6])
newton_fwd_F = np.zeros([4, 4], dtype=float)

newton_fwd_F[:, 0] = np.array([23.5492, 25.3913, 26.8224, 27.4589])

newton_fwd_F = newtons_fwd_diff(newton_fwd_F, newton_fwd_x)
ans_q2 = np.diagonal(newton_fwd_F[1:, 1:])
print('[%.15f, %.14f, %.14f]' % (ans_q2[0], ans_q2[1], ans_q2[2]), end="\n\n")


# Question 3
approx_w = 7.3

ans_q3 = polynomial_approx(newton_fwd_F, newton_fwd_x, approx_w)
print(ans_q3, end="\n\n")


# Question 4
hermite_x = np.array([3.6, 3.8, 3.9])
hermite_f = np.array([1.675, 1.436, 1.318])
hermite_m = np.array([-1.195, -1.188, -1.182])

# create table
hermite_table = np.zeros([6, 6])
hermite_table[:, 0] = np.repeat(hermite_x, 2)
hermite_table[:, 1] = np.repeat(hermite_f, 2)

for i in range(len(hermite_m)):
    hermite_table[1 + (i * 2)][2] = hermite_m[i]

ans_q4 = hermite_interpolation(hermite_table)
print(ans_q4, end="\n\n")


# Question 5
cubic_spline_x = np.array([2, 5, 8, 10])
cubic_spline_f = np.array([3, 5, 7, 9])

cubic_spline_table = np.zeros([4, 4], dtype=float)

ans_q5_a, ans_q5_b, ans_q5_c = cubic_spline_interpolation(cubic_spline_table, cubic_spline_x, cubic_spline_f)
print(ans_q5_a, end="\n\n")
print(ans_q5_b, end="\n\n")
print(ans_q5_c, end="\n\n")
