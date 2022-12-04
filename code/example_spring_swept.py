import numpy as np
from scipy import optimize
import copy
import lst
import matplotlib.pyplot as plt
import example_spring_setting as dyn_setting
from plot_packages import *
import niceplots

# niceplots.setRCParams()


# Define file names to load / save data
filename_con = "contour_constraint_spring.dat"
filename_obj = "contour_objective_spring.dat"
filename_hist = "opt_hist_spring.dat"

isTraining = False

Nx1 = 30
Nx2 = 30
x1_arr = np.linspace(2.0, 5.0, Nx1)
x2_arr = np.linspace(3.0, 6.0, Nx2)

if isTraining:
    win = np.zeros(4)

    obj_arr = np.zeros((Nx1, Nx2))
    stability_arr = np.zeros((Nx1, Nx2))

    for i in range(Nx1):
        for j in range(Nx2):

            print("i, j", i, j)

            x = [x1_arr[i], x2_arr[j]]
            r = lambda w: dyn_setting.f_res(w, x)

            sol = optimize.newton_krylov(r, win, f_tol=1e-10)
            win = sol
            pres_pw = dyn_setting.f_pres_pw(sol, x)

            eig = np.max(np.real(np.linalg.eig(pres_pw)[0]))

            obj_arr[i, j] = dyn_setting.f_obj(sol, x)
            stability_arr[i, j] = eig

    np.savetxt(filename_obj, obj_arr)
    np.savetxt(filename_con, stability_arr)

    quit()
else:
    obj_arr = np.loadtxt(filename_obj)
    stability_arr = np.loadtxt(filename_con)

# Optimization path
x_path = np.loadtxt(filename_hist)
# Optimal solution
x_opt = [2.815165e00, 4.718862e00]

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
X1_arr, X2_arr = np.meshgrid(x1_arr, x2_arr)

# --------------
# Constraint
# --------------

# Countour plot
levels0 = np.arange(-3.0, 3.0, 0.5)
CP0 = ax[0].contour(X1_arr, X2_arr, stability_arr.T, levels0, extend="both", linewidths=2, cmap="coolwarm", zorder=0)

ax[0].clabel(CP0, levels0[1::1], inline=True, fmt="%1.2f", fontsize=14)

ax[0].spines["right"].set_visible(False)
ax[0].spines["top"].set_visible(False)

ax[0].set_xlabel(r"$l_2$", fontsize=20)
ax[0].set_ylabel(r"$l_3$", fontsize=20, rotation=0)

ax[0].yaxis.set_label_coords(-0.1, 0.5)

# Extract the stability boundary
ind_critical = 6
stability_boundary_1 = CP0.collections[ind_critical].get_paths()[0]
stability_boundary_1 = stability_boundary_1.vertices
stability_boundary_2 = CP0.collections[ind_critical].get_paths()[1]
stability_boundary_2 = stability_boundary_2.vertices
stability_boundary = np.concatenate((stability_boundary_1, stability_boundary_2))

# stability_boundary = np.vstack([[5.0, 4.15], stability_boundary])

# Adding optimization paths
# Optimal solution
ax[0].plot(x_opt[0], x_opt[1], "o")
# Path
ax[0].plot(x_path[:, 0], x_path[:, 1], "o", color="w", markersize=10, zorder=3)
ax[0].plot(x_path[:, 0], x_path[:, 1], "o", color=my_brown, markersize=6, zorder=3)
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
ax[0].fill_between(
    stability_boundary[:21, 0], stability_boundary[:21, 1], y2=3, facecolor=my_purple, alpha=0.3, zorder=0
)
ax[0].fill_betweenx(
    stability_boundary[20:, 1], stability_boundary[20:, 0], x2=2, facecolor=my_purple, alpha=0.3, zorder=0
)
ax[0].fill_between(
    [2, stability_boundary[20, 0]],
    [stability_boundary[20, 1], stability_boundary[20, 1]],
    y2=3,
    facecolor=my_purple,
    alpha=0.3,
    zorder=0,
)


ax[0].plot(x_path[0, 0], x_path[0, 1], "s", color="w", markersize=10, zorder=4) 
ax[0].plot(x_path[0, 0], x_path[0, 1], "s", color=my_brown, markersize=8, zorder=4)
ax[0].plot(x_path[-1, 0], x_path[-1, 1], "D", color="w", markersize=10, zorder=4) 
ax[0].plot(x_path[-1, 0], x_path[-1, 1], "D", color=my_brown, markersize=8, zorder=4)


# --------------
# Objective
# --------------
# Countour plot
levels1 = np.arange(2.0, 3.6, 0.1)

CP1 = ax[1].contour(X1_arr, X2_arr, obj_arr.T, levels1, extend="both", linewidths=2, cmap="inferno")

ax[1].clabel(CP1, levels1[1::1], inline=True, fmt="%1.2f", fontsize=14)

ax[1].spines["right"].set_visible(False)
ax[1].spines["top"].set_visible(False)

ax[1].set_xlabel(r"$l_2$", fontsize=20)
# ax[1].set_ylabel(r'$l_3$', fontsize=20, rotation=0)

ax[1].get_yaxis().set_visible(False)

# Adding optimization paths
# Optimal solution
ax[1].plot(x_opt[0], x_opt[1], "o")
# Path
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
# Stability boundary
ax[1].plot(stability_boundary[:, 0], stability_boundary[:, 1], "-", color="k", alpha=0.6, zorder=0)
ax[1].plot(stability_boundary[:, 0], stability_boundary[:, 1], "-", color="k", alpha=0.6, zorder=0)
ax[1].fill_between(
    stability_boundary[:21, 0], stability_boundary[:21, 1], y2=3, facecolor=my_purple, alpha=0.3, zorder=0
)
ax[1].fill_betweenx(
    stability_boundary[20:, 1], stability_boundary[20:, 0], x2=2, facecolor=my_purple, alpha=0.3, zorder=0
)
ax[1].fill_between(
    [2, stability_boundary[20, 0]],
    [stability_boundary[20, 1], stability_boundary[20, 1]],
    y2=3,
    facecolor=my_purple,
    alpha=0.3,
    zorder=0,
)


ax[1].plot(x_path[0, 0], x_path[0, 1], "s", color="w", markersize=10, zorder=4) 
ax[1].plot(x_path[0, 0], x_path[0, 1], "s", color=my_brown, markersize=8, zorder=4)
ax[1].plot(x_path[-1, 0], x_path[-1, 1], "D", color="w", markersize=10, zorder=4) 
ax[1].plot(x_path[-1, 0], x_path[-1, 1], "D", color=my_brown, markersize=8, zorder=4)


plt.tight_layout()
plt.savefig("../R0_journal/figures/contour_spring.pdf")

# plt.show()
