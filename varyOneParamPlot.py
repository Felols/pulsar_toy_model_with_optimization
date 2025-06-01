from matplotlib import pyplot as plt
import numpy as np
from contour import pencil, fan
from helper_functions import get_intensity
from matplotlib import cm


inc_colors = cm.Purples(np.linspace(0.4, 1, 5))
pa_colors = cm.Oranges(np.linspace(0.4, 1, 5))
magco_colors = cm.Greens(np.linspace(0.4, 1, 5))
phs_colors = cm.Blues(np.linspace(0.4, 1, 5))
az_off_colors = cm.Reds(np.linspace(0.4, 1, 5))
inc_off_colors = cm.BuPu(np.linspace(0.4, 1, 5))
gamma_colors = cm.Greys(np.linspace(0.5, 1, 5))

def oneParameterPlot():
    """Plots pulse profiles for N different sets of parameters in one figure. The parameter(s) which increments by
    angle_step or other_step is chosen manually."""
    xarray = np.linspace(0, 1, 32)
    plt.figure()

    inc = np.deg2rad(40)
    pa = np.deg2rad(20)
    magco = np.deg2rad(16)
    phs = np.deg2rad(0)
    az_off = np.deg2rad(0)
    inc_off = np.deg2rad(0)
    r = 3
    gamma = 1

    func = pencil

    N = 5
    angle_step = np.deg2rad(20)
    other_step = 1
    for i in range(N):
        new_inc = inc + i * angle_step
        new_pa = pa + i * angle_step
        new_magco = magco + i * angle_step
        new_phs = phs + i * angle_step
        new_az_off = az_off + i * angle_step
        new_inc_off = inc_off + i * angle_step
        new_r = r + i * other_step
        new_gamma = gamma + i * other_step
        int_adjusted = get_intensity(inc, pa, magco, phs, az_off, inc_off, r, new_gamma, func)
        plt.plot(xarray, int_adjusted, label="γ="+str(round(new_gamma, 3)), color=gamma_colors[i])
    plt.xlabel("Phase", fontsize=15)
    plt.ylabel("Rel. Int.", fontsize=15)
    plt.legend(loc=2)
    plt.show()


def main():
    oneParameterPlot()


if __name__ == '__main__':
    main()