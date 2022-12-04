import numpy as np
from scipy import optimize
import copy
import lst
import matplotlib.pyplot as plt

# ----------
# Bottom level
# ----------

def f_res(w, x):

    # Residual

    w1 = w[0]
    w2 = w[1]

    x1 = x[0]
    x2 = x[1]

    res_1 = (x1 - 1.2 * x2 ** 2) * w1 - w2 + (2.0 * x2 - 1.0) * w1 ** 3 - 0.1
    res_2 = w1 + (x1 - 1.0) * w2 + w2 ** 3
    res = np.zeros(2)
    res[0] = res_1
    res[1] = res_2

    return res

def f_obj(w, x):

    # Objective function

    w1 = w[0]
    w2 = w[1]

    x1 = x[0]
    x2 = x[1]

    obj = 0.3 * (1.0 - x1) ** 2 + 0.5 * (x2 - 0.5) ** 2 + w1 ** 2 + 3 * w2

    return obj

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

    pres_pw[0, 0] = (x1 - 1.2 * x2 ** 2) + (2.0 * x2 - 1.0) * 3.0 * w1 ** 2
    pres_pw[0, 1] = - 1.0
    pres_pw[1, 0] = 1.0
    pres_pw[1, 1] = (x1 - 1.0) + 3.0 * w2 ** 2

    return pres_pw

def f_pres_pw_T(w, x, isComplex = False):

    pres_pw_T = f_pres_pw(w, x, isComplex = isComplex)

    pres_pw_T = pres_pw_T.T

    return pres_pw_T

def f_pobj_pw(w, x):

    # Objective function

    w1 = w[0]
    w2 = w[1]

    x1 = x[0]
    x2 = x[1]

    pobj_pw = np.zeros(2)
    pobj_pw[0] = 2.0 * w1
    pobj_pw[1] = 3.0

    return pobj_pw

def f_pobj_px(w, x):

    # Objective function

    w1 = w[0]
    w2 = w[1]

    x1 = x[0]
    x2 = x[1]

    pobj_px = np.zeros(2)
    pobj_px[0] = 0.3 * 2 * (1.0 - x1) * (- 1.0)
    pobj_px[1] = 0.5 * 2 * (x2 - 0.5)

    return pobj_px

def f_pres_px(w, x):

    # p res / p x

    w1 = w[0]
    w2 = w[1]

    x1 = x[0]
    x2 = x[1]

    f1 = (x1 - 1.2 * x2 ** 2) * w1 - w2 + (2.0 * x2 - 1.0) * w1 ** 3 - 0.1
    f2 = w1 + (x1 - 1.0) * w2 + w2 ** 3

    pres_px = np.zeros((2, 2))

    pres_px[0, 0] = w1
    pres_px[0, 1] = - 2.4 * x2 * w1 + 2.0 * w1 ** 3

    pres_px[1, 0] = w2
    pres_px[1, 1] = 0.0

    return pres_px

def f_p2res_pw2(w, x):

    # p^2 res / p w^2

    w1 = w[0]
    w2 = w[1]

    x1 = x[0]
    x2 = x[1]

    p2res_pw2 = np.zeros((2, 2, 2))

    p2res_pw2[0, 0, 0] = (2.0 * x2 - 1.0) * 6 * w1
    p2res_pw2[1, 1, 1] = 6 * w2

    return p2res_pw2

def f_p2res_pw2_T(w, x, pres_pw_bar):

    # (p^2 res / p w^2)^T wbar
    # NOT VERIFIED

    p2res_pw2 = f_p2res_pw2(w, x)

    wbar = np.zeros(2)
    for i in range(2):
        wbar[i] = np.trace(p2res_pw2[:, :, i].T.dot(pres_pw_bar))

    return wbar

def f_p2res_pwpx(w, x):

    # p^2 res / p w p x

    w1 = w[0]
    w2 = w[1]

    x1 = x[0]
    x2 = x[1]

    p2res_pwpx = np.zeros((2, 2, 2))

    p2res_pwpx[0, 0, 0] = 1.0
    p2res_pwpx[0, 0, 1] = - 1.2 * 2 * x2 + 2.0 * 3.0 * w1 ** 2

    p2res_pwpx[1, 1, 0] = 1.0

    return p2res_pwpx

def f_p2res_pwpx_T(w, x, jacbar):

    # (p^2 res / p w p x)^T pres_pw_bar
    # NOT VERIFIED

    p2res_pwpx = f_p2res_pwpx(w, x)

    xbar = np.zeros(2)
    for i in range(2):
        xbar[i] = np.trace(p2res_pwpx[:, :, i].T.dot(jacbar))

    return xbar

# ----------
# Top level
# ----------

def f_LST(v, x):

    '''
        Linear stability function.
    '''

    return v[-2]

def f_pLST_pv(v, x):

    ndof = np.shape(v)[0]
    pLST_pv = np.zeros(ndof)
    pLST_pv[-2] = 1.0

    return pLST_pv

def f_loss(v, x):

    '''
        Mode loss function.
    '''

    v_tar_r = np.zeros(2)
    v_tar_r[0] = np.sqrt(3) / 2
    v_tar_i = np.zeros(2)
    v_tar_i[1] = 1.0 / 2.0

    v_r = v[0 : 2]
    v_i = v[2 : 4]

    print("v_r", v_r)

    loss = (v_r - v_tar_r).dot(v_r - v_tar_r) + (v_i - v_tar_i).dot(v_i - v_tar_i)

    return loss

def f_ploss_pv(v, x):

    v_tar_r = np.zeros(2)
    v_tar_r[0] = np.sqrt(3) / 2
    v_tar_i = np.zeros(2)
    v_tar_i[1] = 1.0 / 2.0

    v_r = v[0 : 2]
    v_i = v[2 : 4]

    ploss_pv = np.zeros(6)

    ploss_pv[0 : 2] = 2 * (v_r - v_tar_r)
    ploss_pv[2 : 4] = 2 * (v_i - v_tar_i)

    return ploss_pv
