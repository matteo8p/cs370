import numpy as np
import random


def ok(imp, par, tolerance=1.e-6):
    a, b = imp
    p, q = par
    if a.ndim == 1:
        a = np.expand_dims(a, axis=1)
    if q.ndim == 1:
        q = np.expand_dims(q, axis=1)
    d = a.shape[0]
    n = d + 1
    alpha = np.random.randn(d - 1, n)
    points = np.outer(p, np.ones(n)) + np.dot(q, alpha)
    residuals = np.dot(a.T, points) - b
    max_residual = np.max(residuals)
    return max_residual < tolerance


def random_implicit_hyperplane(d):
    tilde_a = np.random.randn(d)
    while np.linalg.norm(tilde_a) < 0.001:
        tilde_a = np.random.randn(d)
    tilde_b = np.random.randn()
    return tilde_a, tilde_b


def random_parametric_hyperplane(d):
    tilde_p = np.random.randn(d)
    tilde_q = np.random.randn(d, d-1)
    while np.linalg.matrix_rank(tilde_q, tol=0.001) < d - 1:
        tilde_q = np.random.randn(d, d - 1)
    return tilde_p, tilde_q


def random_hyperplanes(n):
    hs = []
    for k in range(n):
        rep = random.choice(('i', 'p'))
        d = np.random.randint(low=2, high=10)
        h = random_implicit_hyperplane(d) if rep == 'i' \
            else random_parametric_hyperplane(d)
        hs.append((rep, h))
    return hs


def check_representations(i2p, p2i, n=100):
    test_hyperplanes = random_hyperplanes(n)
    good = True
    for h_type, h_par in test_hyperplanes:
        if h_type == 'i':
            imp = h_par
            par = i2p(*imp)
        else:
            par = h_par
            imp = p2i(*par)
        if not ok(imp, par):
            good = False
            break
    return good
