import numpy as np
from scipy import optimize
import copy
import lst
import matplotlib.pyplot as plt
import example_nonhomogenuous_setting as dyn_setting
from plot_packages import *
import niceplots

niceplots.setRCParams()

# optimization path
filename = "opt_hist.dat"
x_path = np.loadtxt(filename)

ndof = 2

Nx1 = 30
Nx2 = 30
x1_arr = np.linspace(0, 1, Nx1)
x2_arr = np.linspace(0, 1, Nx2)
obj_arr = np.zeros((Nx1, Nx2))
stability_arr = np.zeros((Nx1, Nx2))

for i in range(Nx1):
    for j in range(Nx2):

        print("i, j", i, j)

        x = [x1_arr[i], x2_arr[j]]

        nx = len(x)
        ndof = 2

        f_int_dict = {"obj": dyn_setting.f_obj}
        f_pint_pw_dict = {"obj": dyn_setting.f_pobj_pw}
        f_pint_px_dict = {"obj": dyn_setting.f_pobj_px}

        f_int_top_dict = {"LST": dyn_setting.f_LST}
        f_pint_top_pv_dict = {"LST": dyn_setting.f_pLST_pv}

        lst_obj = lst.lst(
            ndof,
            x,
            dyn_setting.f_res,
            f_int_dict,
            dyn_setting.f_pres_pw,
            dyn_setting.f_pres_px,
            f_pint_pw_dict,
            f_pint_px_dict,
            dyn_setting.f_p2res_pw2_T,
            dyn_setting.f_p2res_pwpx_T,
            f_int_top_dict,
            f_pint_top_pv_dict,
        )
        lst_obj.solve()

        LST = lst_obj.compute_int("LST")
        obj = lst_obj.compute_int("obj")

        obj_arr[i, j] = obj
        stability_arr[i, j] = LST

x_opt = [5.538193e-01, 5.357487e-01]

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
X1_arr, X2_arr = np.meshgrid(x1_arr, x2_arr)

# ====================
# First plot: constraint
# ====================

levels0 = np.arange(-1.0, 1.0, 0.1)
CP0 = ax[0].contour(X1_arr, X2_arr, stability_arr.T, levels0, extend="both", linewidths=2, cmap="coolwarm", zorder=0)

ax[0].clabel(CP0, levels0[1::1], inline=True, fmt="%1.2f", fontsize=14)

ax[0].spines["right"].set_visible(False)
ax[0].spines["top"].set_visible(False)

ax[0].set_xlabel(r"$x_1$", fontsize=20)
ax[0].set_ylabel(r"$x_2$", fontsize=20, rotation=0)

# Extract the stability boundary
ind_critical = 9
print("CP0.collections[ind_critical].get_paths()", CP0.collections[ind_critical].get_paths())
stability_boundary_1 = CP0.collections[ind_critical].get_paths()[0]
stability_boundary_1 = stability_boundary_1.vertices
stability_boundary_2 = CP0.collections[ind_critical].get_paths()[1]
stability_boundary_2 = stability_boundary_2.vertices
stability_boundary = np.concatenate((stability_boundary_1, stability_boundary_2))


# Adding optimization paths
# Optimal solution
# ax[0].plot(x_opt[0], x_opt[1], "o")
# Path
ax[0].plot(x_path[:, 0], x_path[:, 1], "o", color="w", markersize=10, zorder=3)
ax[0].plot(x_path[:, 0], x_path[:, 1], "o", color=my_brown, markersize=6, zorder=3)

ax[0].plot(x_path[0, 0], x_path[0, 1], "s", color="w", markersize=10, zorder=4) 
ax[0].plot(x_path[0, 0], x_path[0, 1], "s", color=my_brown, markersize=8, zorder=4)
ax[0].plot(x_path[-1, 0], x_path[-1, 1], "D", color="w", markersize=10, zorder=4) 
ax[0].plot(x_path[-1, 0], x_path[-1, 1], "D", color=my_brown, markersize=8, zorder=4)

# Add arrow to the path
ax[0].quiver(
    x_path[:-1, 0],
    x_path[:-1, 1],
    x_path[1:, 0] - x_path[:-1, 0],
    x_path[1:, 1] - x_path[:-1, 1],
    color=my_brown,
    scale_units="xy",
    angles="xy",
    scale=1,
    zorder=2,
)

# Stability boundary
ax[0].plot(stability_boundary[:, 0], stability_boundary[:, 1], "-", color="k", alpha=0.6, zorder=0)
ax[0].fill_between(stability_boundary[:, 0], stability_boundary[:, 1], y2=0, color=my_purple, alpha=0.3, zorder=0)

# ====================
# Second plot: obj
# ====================
levels0 = np.arange(-0.4, 0.4, 0.05)

CP1 = ax[1].contour(X1_arr, X2_arr, obj_arr.T, levels0, extend="both", linewidths=2, cmap="inferno")

ax[1].clabel(CP1, levels0[1::1], inline=True, fmt="%1.2f", fontsize=14)

ax[1].spines["right"].set_visible(False)
ax[1].spines["top"].set_visible(False)

ax[1].set_xlabel(r"$x_1$", fontsize=20)
ax[1].set_ylabel(r"$x_2$", fontsize=20, rotation=0)

ax[1].plot(x_path[:, 0], x_path[:, 1], "o", color="w", markersize=10, zorder=3)
ax[1].plot(x_path[:, 0], x_path[:, 1], "o", color=my_brown, markersize=6, zorder=3)
# Add arrow to the path
ax[1].quiver(
    x_path[:-1, 0],
    x_path[:-1, 1],
    x_path[1:, 0] - x_path[:-1, 0],
    x_path[1:, 1] - x_path[:-1, 1],
    color=my_brown,
    scale_units="xy",
    angles="xy",
    scale=1,
    zorder=2,
)

ax[1].get_yaxis().set_visible(False)

ax[1].plot(x_path[0, 0], x_path[0, 1], "s", color="w", markersize=10, zorder=4) 
ax[1].plot(x_path[0, 0], x_path[0, 1], "s", color=my_brown, markersize=8, zorder=4)
ax[1].plot(x_path[-1, 0], x_path[-1, 1], "D", color="w", markersize=10, zorder=4) 
ax[1].plot(x_path[-1, 0], x_path[-1, 1], "D", color=my_brown, markersize=8, zorder=4)

# Stability boundary
ax[1].plot(stability_boundary[:, 0], stability_boundary[:, 1], "-", color="k", alpha=0.6, zorder=0)
ax[1].fill_between(stability_boundary[:, 0], stability_boundary[:, 1], y2=0, color=my_purple, alpha=0.3, zorder=0)

plt.tight_layout()
plt.savefig("../R0_journal/figures/contour.pdf")

plt.show()
