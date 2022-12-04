import numpy as np
import copy

def f_res(w, x):

    # Take in the state variables
    w1dot = w[0]
    w2dot = w[1]
    w1 = w[2]
    w2 = w[3]

    # Take in the design variables
    l2 = x[0]
    l3 = x[1]

    # Fill the rest
    l = 10.0
    l1 = l - l2

    # The l3 parameter
    d = 7.5
    h = 5.0

    # Damper
    c = 10.0

    # Stiffness
    k = 100

    k1 = k
    k2 = k
    k3 = 3 * k

    # Mass
    m = 1.0
    g = 9.8

    # Elongation
    delta_l1 = np.sqrt((l1 + w1) ** 2 + w2 ** 2) - l1
    delta_l2 = np.sqrt((l2 - w1) ** 2 + w2 ** 2) - l2
    delta_l3 = np.sqrt((d - l1 - w1) ** 2 + (h - w2) ** 2) - l3

    delta_l1_dot = 1.0 / (2.0 * np.sqrt((l1 + w1) ** 2 + w2 ** 2)) * (2.0 * (l1 + w1) * w1dot + 2.0 * w2dot)

    # Angle
    theta1 = np.arctan(w2 / (l1 + w1))
    theta2 = np.arctan(w2 / (l2 - w1))
    theta3 = np.arctan((d - w1) / (h - w2))

    # Internal force
    f1 = k1 * delta_l1 + delta_l1_dot * c
    f2 = k2 * delta_l2
    f3 = k3 * delta_l3

    # Total force
    fx = - f1 * np.cos(theta1) + f2 * np.cos(theta2) + f3 * np.sin(theta3)
    fy = - f1 * np.sin(theta1) - f2 * np.sin(theta2) + f3 * np.cos(theta3) - m * g

    res = np.zeros(4)
    res[0] = fx
    res[1] = fy
    res[2] = w1dot
    res[3] = w2dot

    return res

def f_obj(w, x):

    # Objective function

    x1 = x[0]
    x2 = x[1]

    w1dot = w[0]
    w2dot = w[1]
    w1 = w[2]
    w2 = w[3]

    weight = 0.5
    obj = weight * np.sqrt(w1 ** 2 + w2 ** 2) + (1 - weight) * x2

    return obj

def f_pres_pw(w, x):

    h = 1e-4

    pres_pw = np.zeros((4, 4))
    for j in range(4):
        wpp = copy.deepcopy(w)
        wpm = copy.deepcopy(w)

        wpp[j] += h
        wpm[j] -= h

        pres_pw[:, j] = (f_res(wpp, x) - f_res(wpm, x)) / (2 * h)

    return pres_pw

def f_pres_pw_T(w, x, isComplex = False):

    pres_pw_T = f_pres_pw(w, x, isComplex = isComplex)

    pres_pw_T = pres_pw_T.T

    return pres_pw_T

def f_pres_px(w, x):

    h = 1e-4

    pres_px = np.zeros((4, 2))
    for j in range(2):
        xpp = copy.deepcopy(x)
        xpm = copy.deepcopy(x)

        xpp[j] += h
        xpm[j] -= h

        pres_px[:, j] = (f_res(w, xpp) - f_res(w, xpm)) / (2 * h)

    return pres_px

def f_pobj_pw(w, x):

    # Objective function

    w1dot = w[0]
    w2dot = w[1]
    w1 = w[2]
    w2 = w[3]

    pobj_pw = np.zeros(4)
    # for i in range(4):
    #     pobj_pw[i] = 1.0
    
    pobj_pw[2] = 2 * w1
    pobj_pw[3] = 2 * w2

    weight = 0.5
    pobj_pw[2] = weight * (1.0 / 2.0) * (1.0 / np.sqrt(w1 ** 2 + w2 ** 2)) * (2 * w1)
    pobj_pw[3] = weight * (1.0 / 2.0) * (1.0 / np.sqrt(w1 ** 2 + w2 ** 2)) * (2 * w2)
    
    return pobj_pw

def f_pobj_px(w, x):

    # Objective function

    pobj_px = np.zeros(2)

    weight = 0.5
    pobj_px[1] = (1 - weight)

    return pobj_px



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
