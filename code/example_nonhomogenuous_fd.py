import numpy as np
from scipy import optimize
import copy
import lst
import matplotlib.pyplot as plt
import example_nonhomogenuous_setting as dyn_setting
from plot_packages import *

isPartial = False
isTotal = True
isLowAdjoint = False
isFull = True
isLST = True

if isPartial:
    np.random.seed(0)
    w = np.random.rand(2)
    x = np.random.rand(2)

    # f_pobj_pw
    epsilon = 1e-6

    pobj_pw_FD = np.zeros(2)

    obj_0 = dyn_setting.f_obj(w, x)

    for k in range(2):

        w_p = copy.deepcopy(w)
        w_p[k] += epsilon
        obj_p = dyn_setting.f_obj(w_p, x)

        pobj_pw_FD[k] = (obj_p - obj_0) / epsilon

    pobj_pw_analytic = dyn_setting.f_pobj_pw(w, x)

    print("pobj_pw_FD", pobj_pw_FD)
    print("pobj_pw_analytic", pobj_pw_analytic)
    print("Error", pobj_pw_FD - pobj_pw_analytic)

    # f_pobj_px
    epsilon = 1e-6

    pobj_px_FD = np.zeros(2)

    obj_0 = dyn_setting.f_obj(w, x)

    for k in range(2):

        x_p = copy.deepcopy(x)
        x_p[k] += epsilon
        obj_p = dyn_setting.f_obj(w, x_p)

        pobj_px_FD[k] = (obj_p - obj_0) / epsilon

    pobj_px_analytic = dyn_setting.f_pobj_px(w, x)

    print("pobj_px_FD", pobj_px_FD)
    print("pobj_px_analytic", pobj_px_analytic)
    print("Error", pobj_px_FD - pobj_px_analytic)

    # f_pres_pw
    epsilon = 1e-6

    pres_pw_FD = np.zeros((2, 2))

    res_0 = dyn_setting.f_res(w, x)

    for k in range(2):

        w_p = copy.deepcopy(w)
        w_p[k] += epsilon
        res_p = dyn_setting.f_res(w_p, x)

        pres_pw_FD[:, k] = (res_p - res_0) / epsilon

    pres_pw_analytic = dyn_setting.f_pres_pw(w, x)

    print("pres_pw_FD", pres_pw_FD)
    print("pres_pw_analytic", pres_pw_analytic)
    print("Error", pres_pw_FD - pres_pw_analytic)

    # f_pres_px
    epsilon = 1e-6

    pres_px_FD = np.zeros((2, 2))

    res_0 = dyn_setting.f_res(w, x)

    for k in range(2):

        x_p = copy.deepcopy(x)
        x_p[k] += epsilon
        res_p = dyn_setting.f_res(w, x_p)

        pres_px_FD[:, k] = (res_p - res_0) / epsilon

    pres_px_analytic = dyn_setting.f_pres_px(w, x)

    print("pres_px_FD", pres_px_FD)
    print("pres_px_analytic", pres_px_analytic)
    print("Error", pres_px_FD - pres_px_analytic)

    # f_p2res_pw2
    epsilon = 1e-6

    p2res_pw2_FD = np.zeros((2, 2, 2))

    pres_pw_0 = dyn_setting.f_pres_pw(w, x)

    for k in range(2):

        w_p = copy.deepcopy(w)
        w_p[k] += epsilon
        pres_pw_p = dyn_setting.f_pres_pw(w_p, x)

        p2res_pw2_FD[:, :, k] = (pres_pw_p - pres_pw_0) / epsilon

    p2res_pw2_analytic = dyn_setting.f_p2res_pw2(w, x)

    print("p2res_pw2_FD", p2res_pw2_FD)
    print("p2res_pw2_analytic", p2res_pw2_analytic)
    print("Error", p2res_pw2_FD - p2res_pw2_analytic)

    # pjacf_p2res_pwpxpx
    epsilon = 1e-6

    p2res_pwpx_FD = np.zeros((2, 2, 2))

    pres_pw_0 = dyn_setting.f_pres_pw(w, x)

    for k in range(2):

        x_p = copy.deepcopy(x)
        x_p[k] += epsilon
        pres_pw_p = dyn_setting.f_pres_pw(w, x_p)

        p2res_pwpx_FD[:, :, k] = (pres_pw_p - pres_pw_0) / epsilon

    p2res_pwpx_analytic = dyn_setting.f_p2res_pwpx(w, x)

    print("p2res_pwpx_FD", p2res_pwpx_FD)
    print("p2res_pwpx_analytic", p2res_pwpx_analytic)
    print("Error", p2res_pwpx_FD - p2res_pwpx_analytic)


if isTotal:
    if ((isLowAdjoint) and (not isFull)):

        x = [0.3, 0.7]

        nx = len(x)
        ndof = 2
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

            ndof = 2
            nonlinear_obj = lst.nonlinear_eqn(ndof, xp, dyn_setting.f_res, f_int_dict, dyn_setting.f_pres_pw, dyn_setting.f_pres_px, f_pint_pw_dict, f_pint_px_dict)
            nonlinear_obj.solve()

            obj_p = nonlinear_obj.compute_int(key)

            total_der_FD[i] = (obj_p - obj_0) / epsilon

        print("total_der_FD", total_der_FD)
        print("total_der", total_der)
        print("Error", total_der_FD - total_der)


if isTotal:
    if ((not isLowAdjoint) and (not isFull)):

        nx = 2
        ndof = 2

        np.random.seed(0)
        w = np.random.rand(ndof)
        x = np.random.rand(nx)

        key = "LST"
        f_int_top_dict = {key: dyn_setting.f_LST}
        f_pint_top_pv_dict = {key: dyn_setting.f_pLST_pv}

        eig_eqn_obj = lst.eig_eqn(ndof, x, w, dyn_setting.f_pres_pw, f_int_top_dict, f_pint_top_pv_dict)
        eig_eqn_obj.solve()
        eig_eqn_obj.solve_adjoint(key)

        eig_eqn_obj.compute_total_der(key)
        [phi1, qr, phi2, qi] = eig_eqn_obj.get_total_der(key)

        Jbar = - np.outer(phi1, qr) - np.outer(phi2, qi)

        jac = dyn_setting.f_pres_pw(w, x)
        eigval_0 = np.max(np.real(np.linalg.eig(jac)[0]))
        Jbar_FD = np.zeros((2, 2))

        epsilon = 1e-6
        for i in range(2):
            for j in range(2):
                jac_p = copy.deepcopy(jac)
                jac_p[i, j] += epsilon

                eigval_p = np.max(np.real(np.linalg.eig(jac_p)[0]))

                Jbar_FD[i, j] = (eigval_p - eigval_0) / epsilon

        print("Jbar", Jbar)
        print("Jbar_FD", Jbar_FD)
        print("Error", Jbar - Jbar_FD)

if isTotal:
    if isFull:
        if isLST:
            
            epsilon = 1e-6
            
            x = [0.3, 0.7]
            nx = len(x)
            ndof = 2

            # key = "LST"
            key = "loss"

            f_int_dict = {}
            f_pint_pw_dict = {}
            f_pint_px_dict = {}

            if (key == "LST"):
                f_int_top_dict = {key: dyn_setting.f_LST}
                f_pint_top_pv_dict = {key: dyn_setting.f_pLST_pv}
            elif (key == "loss"):
                f_int_top_dict = {key: dyn_setting.f_loss}
                f_pint_top_pv_dict = {key: dyn_setting.f_ploss_pv}

            lst_obj = lst.lst(ndof, x, dyn_setting.f_res, f_int_dict, dyn_setting.f_pres_pw, dyn_setting.f_pres_px, f_pint_pw_dict, f_pint_px_dict, dyn_setting.f_p2res_pw2_T, dyn_setting.f_p2res_pwpx_T, f_int_top_dict, f_pint_top_pv_dict, f_pint_top_px_dict = None, f_pint_top_pJacobian_dict = None, useAnalyticHessian = False)
            lst_obj.solve()

            LST_0 = lst_obj.compute_int(key)
            lst_obj.compute_total_der(key)
            LST_der = lst_obj.get_total_der(key)

            LST_der_FD = np.zeros(nx)
            for i in range(nx):
                xp = copy.deepcopy(x)
                xp[i] += epsilon

                lst_obj = lst.lst(ndof, xp, dyn_setting.f_res, f_int_dict, dyn_setting.f_pres_pw, dyn_setting.f_pres_px, f_pint_pw_dict, f_pint_px_dict, dyn_setting.f_p2res_pw2_T, dyn_setting.f_p2res_pwpx_T, f_int_top_dict, f_pint_top_pv_dict, f_pint_top_px_dict = None, f_pint_top_pJacobian_dict = None)
                lst_obj.solve()
                print("v", lst_obj.get_v())

                LST_p = lst_obj.compute_int(key)

                LST_der_FD[i] = (LST_p - LST_0) / epsilon

            print("LST_der", LST_der)
            print("LST_der_FD", LST_der_FD)
            print("Error", np.true_divide(LST_der - LST_der_FD, LST_der_FD))

if isTotal:
    if isFull:
        if (not isLST):

            epsilon = 1e-6
            
            key = "obj"
            x = [0.3, 0.7]
            nx = len(x)
            ndof = 2

            f_int_dict = {"obj": dyn_setting.f_obj}
            f_pint_pw_dict = {"obj": dyn_setting.f_pobj_pw}
            f_pint_px_dict = {"obj": dyn_setting.f_pobj_px}

            f_int_top_dict = {"LST": dyn_setting.f_LST}
            f_pint_top_pv_dict = {"LST": dyn_setting.f_pLST_pv}

            lst_obj = lst.lst(ndof, x, dyn_setting.f_res, f_int_dict, dyn_setting.f_pres_pw, dyn_setting.f_pres_px, f_pint_pw_dict, f_pint_px_dict, dyn_setting.f_p2res_pw2_T, dyn_setting.f_p2res_pwpx_T, f_int_top_dict, f_pint_top_pv_dict)
            lst_obj.solve()

            LST_0 = lst_obj.compute_int(key)
            lst_obj.compute_total_der(key)
            LST_der = lst_obj.get_total_der(key)

            LST_der_FD = np.zeros(nx)
            for i in range(nx):
                xp = copy.deepcopy(x)
                xp[i] += epsilon

                lst_obj = lst.lst(ndof, xp, dyn_setting.f_res, f_int_dict, dyn_setting.f_pres_pw, dyn_setting.f_pres_px, f_pint_pw_dict, f_pint_px_dict, dyn_setting.f_p2res_pw2_T, dyn_setting.f_p2res_pwpx_T, f_int_top_dict, f_pint_top_pv_dict)
                lst_obj.solve()

                LST_p = lst_obj.compute_int(key)

                LST_der_FD[i] = (LST_p - LST_0) / epsilon

            print("LST_der", LST_der)
            print("LST_der_FD", LST_der_FD)
            print("Error", np.true_divide(LST_der - LST_der_FD, LST_der_FD))
