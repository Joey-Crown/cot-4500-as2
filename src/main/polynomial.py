# This package contains all the functions necessary to solve the problems in the assignment
import numpy as np


# performs Neville's method of polynomial interpolation
# returns the interpolated value of w
def nevilles_method(q, x, w):
    for i in range(1, len(q)):
        for j in range(1, i + 1):
            t1 = (w - x[i - j]) * q[i][j - 1]
            t2 = (w - x[i]) * q[i - 1][j - 1]
            q[i][j] = (t1 - t2) / (x[i] - x[i - j])
    return q[len(q) - 1][len(q) - 1]


# Newton's forward method for polynomial approximation
# returns an array
def newtons_fwd_diff(f, x):
    for i in range(1, len(f)):
        for j in range(1, i + 1):
            f[i][j] = (f[i][j - 1] - f[i - 1][j - 1]) / (x[i] - x[i - j])
    return f


# takes the polynomial approximation found with the newton's forward method
# computes and returns f(w)
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


# uses the divided difference method to compute the Hermite polynomial approximation of f
# returns a completed table of derivatives
def hermite_interpolation(q):
    for i in range(2, len(q)):
        for j in range(2, i + 2):
            if j >= len(q[i]) or q[i][j] != 0:
                continue

            t1 = q[i][j - 1] - q[i - 1][j - 1]
            t2 = q[i][0] - q[i - j + 1][0]
            q[i][j] = t1 / t2
    return q


# interpolates a polynomial using the cubic spline method
# returns the matrix A, vector b, and vector x such that Ax = b
def cubic_spline_interpolation(a, x, f):
    n = len(a)
    h = np.zeros(n - 1)
    for i in range(0, n - 1):
        h[i] = x[i + 1] - x[i]

    a[0][0] = 1
    a[n - 1][n - 1] = 1
    for i in range(1, n - 1):
        for j in range(i - 1, i + 2):
            if j == i:
                a[i][j] = 2 * (h[i - 1] + h[i])
            else:
                a[i][j] = h[i - 1] if j < i else h[i]

    b = np.zeros(n)
    for i in range(1, n - 1):
        t1 = (3 / h[i]) * (f[i + 1] - f[i])
        t2 = (3 / h[i - 1]) * (f[i] - f[i - 1])
        b[i] = t1 - t2

    c, l, u, z, p, d = np.zeros([6, n])
    l[0] = 1
    for i in range(1, n - 1):
        l[i] = 2 * (x[i + 1] - x[i - 1]) - (h[i - 1] * u[i - 1])
        u[i] = h[i] / l[i]
        z[i] = (b[i] - (h[i-1] * z[i - 1])) / l[i]
    l[n - 1] = 1

    for j in range(n - 2, 0, -1):
        c[j] = z[j] - (u[j] * c[j + 1])
        p[j] = ((f[j + 1] - f[j]) / h[j]) - (h[j] * (c[j + 1] + 2 * c[j])) / 3
        d[j] = (c[j + 1] - c[j]) / (3 * h[j])

    return a, b, c

