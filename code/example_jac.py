import numpy as np
from scipy import optimize
import copy
import lst
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib as matplotlib
import example_jac_setting as dyn_setting
from plot_packages import *
import niceplots

niceplots.setRCParams()


x = [0.3, 0.7]
nx = len(x)
ndof = 2

lst_obj = lst.lst(
    ndof, x, dyn_setting.f_res, None, dyn_setting.f_pres_pw, None, None, None, None, None, None, None, None
)
lst_obj.solve()

w = lst_obj.get_w()

np.random.seed(0)
p1 = np.random.rand(2)
p2 = np.random.rand(2)
print("p1", p1)
print("p2", p2)

# Analytic
pres_pw_bar = np.outer(p1, p2)
wbar_analytic = dyn_setting.f_p2res_pw2_T(w, x, pres_pw_bar)

# FD
def pres_pw_T_p1_p2_FD(w, x, p1, p2, epsilon, isCentral=False):

    if not isCentral:
        pres_pwp = dyn_setting.f_pres_pw_T(w + epsilon * p2, x)
        pres_pw = dyn_setting.f_pres_pw_T(w, x)

        term_1 = pres_pwp.dot(p1)
        term_2 = pres_pw.dot(p1)

        der = (term_1 - term_2) / epsilon
    else:
        pres_pwpp = dyn_setting.f_pres_pw_T(w + epsilon * p2, x)
        pres_pwpm = dyn_setting.f_pres_pw_T(w - epsilon * p2, x)

        term_1 = pres_pwpp.dot(p1)
        term_2 = pres_pwpm.dot(p1)

        der = (term_1 - term_2) / (2 * epsilon)

    return der


epsilon_arr = pow(10, np.linspace(1, -200, 100))
error_FD_arr = np.zeros(len(epsilon_arr))
error_CD_arr = np.zeros(len(epsilon_arr))
for i in range(len(epsilon_arr)):
    epsilon = epsilon_arr[i]

    wbar_FD = pres_pw_T_p1_p2_FD(w, x, p1, p2, epsilon)
    wbar_CD = pres_pw_T_p1_p2_FD(w, x, p1, p2, epsilon, isCentral=True)

    error_FD_arr[i] = np.linalg.norm(wbar_FD - wbar_analytic) / np.linalg.norm(wbar_analytic)
    error_CD_arr[i] = np.linalg.norm(wbar_CD - wbar_analytic) / np.linalg.norm(wbar_analytic)

# CS
def pres_pw_T_p1_p2_CS(w, x, p1, p2, epsilon):

    pres_pwp = dyn_setting.f_pres_pw_T(w + 1j * epsilon * p2, x, isComplex=True)
    term = pres_pwp.dot(p1)

    der = np.imag(term) / epsilon

    return der


error_CS_arr = np.zeros(len(epsilon_arr))
for i in range(len(epsilon_arr)):
    epsilon = epsilon_arr[i]
    wbar_CS = pres_pw_T_p1_p2_CS(w, x, p1, p2, epsilon)

    error_CS_arr[i] = np.linalg.norm(wbar_CS - wbar_analytic) / np.linalg.norm(wbar_analytic)
    if error_CS_arr[i] == 0:
        error_CS_arr[i] = 1e-18

f = plt.figure(figsize=(15, 10))
ax = f.add_subplot(111)
ax.plot(epsilon_arr, error_FD_arr, "-", color=my_blue, linewidth=3, alpha=1)
ax.plot(epsilon_arr, error_FD_arr, "o", color="w", markersize=15)
ax.plot(epsilon_arr, error_FD_arr, "o", color=my_blue, markersize=10, alpha=1)
ax.plot(epsilon_arr, error_CD_arr, "-", color=my_brown, linewidth=3, alpha=1)
ax.plot(epsilon_arr, error_CD_arr, "o", color="w", markersize=15)
ax.plot(epsilon_arr, error_CD_arr, "o", color=my_brown, markersize=10, alpha=1)
ax.plot(epsilon_arr, error_CS_arr, "-", color=my_green, linewidth=3, alpha=1)
ax.plot(epsilon_arr, error_CS_arr, "o", color="w", markersize=15)
ax.plot(epsilon_arr, error_CS_arr, "o", color=my_green, markersize=10, alpha=1)

ax.set_xscale("log")
ax.set_yscale("log")
ax.invert_xaxis()

ax.set_xlabel(r"Step size, $h$", fontsize=30)
ax.set_ylabel(r"Relative error, $\epsilon$", fontsize=30, rotation=0)
ax.yaxis.set_label_coords(-0.2, 0.5)

ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

ax.text(1e-18, 1e-2, r"FDRAD", color=my_blue, fontsize=20, alpha=1)
ax.text(1e-10, 3 * 1e-9, r"CDRAD", color=my_brown, fontsize=20, alpha=1)
ax.text(1e-24, 1e-15, r"CSRAD", color=my_green, fontsize=20, alpha=1)

ax.plot([1e-18, 1e-5],[1e-2, 1e-4], color=my_blue, alpha = 0.5)

ax.set_xlim([1e10, 1e-210])
ax.set_ylim([1e-20, 100])

ax.set_xticks([1e0, 1e-40, 1e-80, 1e-120, 1e-160, 1e-200], [r"$1.0$", r"$10^{-40}$", r"$10^{-80}$", r"$10^{-120}$", r"$10^{-160}$", r"$10^{-200}$"])
ax.set_yticks([1e0, 1e-5, 1e-10, 1e-15, 1e-20], [r"$1.0$", r"$10^{-5}$", r"$10^{-10}$", r"$10^{-15}$", r"$10^{-20}$"])

# niceplots.All()

plt.tight_layout()
plt.savefig("../R0_journal/figures/jac_convergence.pdf")

plt.show()
