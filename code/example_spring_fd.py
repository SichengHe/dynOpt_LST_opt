import numpy as np
from scipy import optimize
import copy
import lst
import matplotlib.pyplot as plt
import example_spring_setting as dyn_setting
from plot_packages import *

if 0:

    # Partial der

    np.random.seed(0)
    w = np.random.rand(4)
    x = np.random.rand(2)

    pres_pw = dyn_setting.f_pres_pw(w, x)
    print("pres_pw", pres_pw)


    epsilon = 1e-6
    pres_pw_FD = np.zeros((4, 4))
    res = dyn_setting.f_res(w, x)
    for i in range(4):
        w_p = copy.deepcopy(w)
        w_p[i] += epsilon

        res_p = dyn_setting.f_res(w_p, x)
        pres_pw_FD[:, i] = (res_p - res) / epsilon

    print("pres_pw_FD", pres_pw_FD)

if 0:

    # Bottom level adjoint

    x = [3.0, 5.0]

    nx = len(x)
    ndof = 4
    f_int_dict = {"obj": dyn_setting.f_obj}
    f_pint_pw_dict = {"obj": dyn_setting.f_pobj_pw}
    f_pint_px_dict = {"obj": dyn_setting.f_pobj_px}
    nonlinear_obj = lst.nonlinear_eqn(ndof, x, dyn_setting.f_res, f_int_dict, dyn_setting.f_pres_pw, dyn_setting.f_pres_px, f_pint_pw_dict, f_pint_px_dict)
    nonlinear_obj.solve()

    key = "obj"
    obj_0 = nonlinear_obj.compute_int(key)

    # Adjoint
    nonlinear_obj.solve_adjoint(key)
    nonlinear_obj.compute_total_der(key)
    total_der = nonlinear_obj.get_total_der(key)

    # FD
    total_der_FD = np.zeros(nx)
    epsilon = 1e-4
    for i in range(nx):
        xp = copy.deepcopy(x)
        xp[i] += epsilon

        nonlinear_obj = lst.nonlinear_eqn(ndof, xp, dyn_setting.f_res, f_int_dict, dyn_setting.f_pres_pw, dyn_setting.f_pres_px, f_pint_pw_dict, f_pint_px_dict)
        nonlinear_obj.solve()

        obj_p = nonlinear_obj.compute_int(key)

        total_der_FD[i] = (obj_p - obj_0) / epsilon

    print("total_der_FD", total_der_FD)
    print("total_der", total_der)
    print("Error", total_der_FD - total_der)

if 0:

    # Total der

    epsilon = 1e-6

    x = [3.0, 5.0]
    nx = len(x)
    ndof = 4

    key = "LST"

    f_int_dict = {}
    f_pint_pw_dict = {}
    f_pint_px_dict = {}

    f_int_top_dict = {key: dyn_setting.f_LST}
    f_pint_top_pv_dict = {key: dyn_setting.f_pLST_pv}

    lst_obj = lst.lst(ndof, x, dyn_setting.f_res, f_int_dict, dyn_setting.f_pres_pw, dyn_setting.f_pres_px, f_pint_pw_dict, f_pint_px_dict, None, None, f_int_top_dict, f_pint_top_pv_dict, f_pint_top_px_dict = None, f_pint_top_pJacobian_dict = None, useAnalyticHessian = False)
    lst_obj.solve()

    LST_0 = lst_obj.compute_int(key)
    lst_obj.compute_total_der(key)
    LST_der = lst_obj.get_total_der(key)

    LST_der_FD = np.zeros(nx)
    for i in range(nx):
        xp = copy.deepcopy(x)
        xp[i] += epsilon

        lst_obj = lst.lst(ndof, xp, dyn_setting.f_res, f_int_dict, dyn_setting.f_pres_pw, dyn_setting.f_pres_px, f_pint_pw_dict, f_pint_px_dict, None, None, f_int_top_dict, f_pint_top_pv_dict, f_pint_top_px_dict = None, f_pint_top_pJacobian_dict = None)
        lst_obj.solve()

        LST_p = lst_obj.compute_int(key)

        LST_der_FD[i] = (LST_p - LST_0) / epsilon

    print("LST_der", LST_der)
    print("LST_der_FD", LST_der_FD)
    print("Error", np.true_divide(LST_der - LST_der_FD, LST_der_FD))

