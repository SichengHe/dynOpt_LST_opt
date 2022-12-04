import numpy as np
from scipy import optimize
import copy
import lst
import matplotlib.pyplot as plt
import example_spring_setting as dyn_setting

x0 = [4.5, 5.5]
nx = len(x0)
ndof = 4

f_int_dict = {"obj": dyn_setting.f_obj}
f_pint_pw_dict = {"obj": dyn_setting.f_pobj_pw}
f_pint_px_dict = {"obj": dyn_setting.f_pobj_px}

f_int_top_dict = {"LST": dyn_setting.f_LST}
f_pint_top_pv_dict = {"LST": dyn_setting.f_pLST_pv}

lst_obj = lst.lst(ndof, x0, dyn_setting.f_res, f_int_dict, dyn_setting.f_pres_pw, dyn_setting.f_pres_px, f_pint_pw_dict, f_pint_px_dict, None, None, f_int_top_dict, f_pint_top_pv_dict, useAnalyticHessian = False)
lst_obj.solve()

# ========================================
#    Optimization problem setup
# ========================================

def objfunc(xdict):

    global lst_obj

    # Extract the design var
    x = xdict["xvars"]

    # Solve the equation
    lst_obj.set_x(x)
    win = lst_obj.get_w()
    lst_obj.solve(win = win)

    # Extract the objective function
    obj = lst_obj.compute_int("obj")

    # Extract the constraint
    LST = lst_obj.compute_int("LST")

    # Set the objective function and con
    funcs = {}
    funcs["obj"] = obj
    funcs["con"] = LST

    # Set failure flag
    fail = False

    return funcs, fail

def sens(xdict, funcs):

    global lst_obj

    # Extract the design variable
    x = xdict["xvars"]
    nx = len(x)

    # Solve the equation
    lst_obj.set_x(x)
    win = lst_obj.get_w()
    lst_obj.solve(win = win)

    # Compute the objective derivative
    lst_obj.compute_total_der("obj")
    dobj_dx = lst_obj.get_total_der("obj")

    # Compute the constraint derivative
    lst_obj.compute_total_der("LST")
    dLST_dx = lst_obj.get_total_der("LST")

    # Set the objective function and con derivative
    x = xdict["xvars"]
    funcsSens = {
        "obj": {
            "xvars": dobj_dx
        },
        "con": {
            "xvars": dLST_dx
        },
    }

    fail = False
    return funcsSens, fail

# ========================================
#    Optimization problem setup
# ========================================

from pyoptsparse import OPT, Optimization

# Optimization Object
optProb = Optimization("Design optimization with LST constraint", objfunc)

# Design Variables
lower = [2.0, 3.0]
upper = [5.0, 6.0]
value = x0
optProb.addVarGroup("xvars", 2, lower=lower, upper=upper, value=value)

# Constraints
lower = [None]
upper = [0]
optProb.addConGroup("con", 1, lower=lower, upper=upper)

# Objective
optProb.addObj("obj")

# Check optimization problem:
print(optProb)

# Optimizer
optimizer = 'snopt'
optOptions = {}
opt = OPT(optimizer, options=optOptions)

# Solution
histFileName = "%s_LCO_Hist.hst" % optimizer

sol = opt(optProb, sens=sens, storeHistory=histFileName)

# Check Solution
print(sol)