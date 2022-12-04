import numpy as np
from scipy import optimize
import copy
import lst
import matplotlib.pyplot as plt

def f_res(w, x):

    # Residual

    w1 = w[0]
    w2 = w[1]

    res_1 = 0.3 * w1 - w2 + w1 ** 3 - 0.2 * np.exp(w1) - 0.1
    res_2 = w1 + 0.2 * w2 + w2 ** 3 - w1 * w2 ** 3
    res = np.zeros(2)
    res[0] = res_1
    res[1] = res_2

    return res

def f_pres_pw(w, x, isComplex = False):

    # Residual Jacobian
    # p res / p w

    w1 = w[0]
    w2 = w[1]

    x1 = x[0]
    x2 = x[1]

    if (not isComplex):
        pres_pw = np.zeros((2, 2))
    else:
        pres_pw = np.zeros((2, 2),dtype=complex)

    pres_pw[0, 0] = 0.3 + 3.0 * w1 ** 2 - 0.2 * np.exp(w1)
    pres_pw[0, 1] = - 1.0
    pres_pw[1, 0] = 1.0 - w2 ** 3
    pres_pw[1, 1] = 0.2 + 3.0 * w2 ** 2 - 3.0 * w1 * w2 ** 2

    return pres_pw

def f_pres_pw_T(w, x, isComplex = False):

    pres_pw_T = f_pres_pw(w, x, isComplex = isComplex)

    pres_pw_T = pres_pw_T.T

    return pres_pw_T

def f_p2res_pw2(w, x):

    # p^2 res / p w^2

    w1 = w[0]
    w2 = w[1]

    p2res_pw2 = np.zeros((2, 2, 2))

    res_1 = 0.3 * w1 - w2 + w1 ** 3 - 0.2 * np.exp(w1) - 0.1
    res_2 = w1 + 0.2 * w2 + w2 ** 3 - w1 * w2 ** 3

    p2res_pw2[0, 0, 0] = 6.0 * w1 - 0.2 * np.exp(w1)
    p2res_pw2[1, 1, 1] = 6 * w2 - w1 * (6 * w2)
    p2res_pw2[1, 1, 0] = - 3 * w2 ** 2
    p2res_pw2[1, 0, 1] = - 3 * w2 ** 2

    return p2res_pw2

def f_p2res_pw2_T(w, x, pres_pw_bar):

    # (p^2 res / p w^2)^T wbar
    # NOT VERIFIED

    p2res_pw2 = f_p2res_pw2(w, x)

    wbar = np.zeros(2)
    for i in range(2):
        wbar[i] = np.trace(p2res_pw2[:, :, i].T.dot(pres_pw_bar))

    return wbar