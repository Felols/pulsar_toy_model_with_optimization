import numpy as np
from helper_functions import get_intensity, weighted_norm, gravity_adjust
from contour import Contour, pencil, fan, combine
from sphere import rotate_vector
import csv
import ast
from matplotlib import pyplot as plt

def main(criteria, row_num):
    """Plots the intensity profiles given from parameters stored in file at specified row

    Args:
        criteria (string): criteria name
        row_num (int): row number
    """
    fit_values = []
    parameters = []
    func = []
    mirror = []
    dipole_force = []

    file = criteria + ".csv"
    
    d1 = np.array([2.08943366e+02, 2.19458930e+02, 2.24270196e+02, 2.21582811e+02,
                    2.11904408e+02, 1.89629016e+02, 1.69920199e+02, 1.25652313e+02,
                    8.50368827e+01, 7.87287160e+01, 6.99192303e+01, 5.68317891e+01,
                    3.12415676e+01, 7.26903682e+01, 6.02122557e+01, 3.28354915e-05,
                    2.07074634e+01, 2.03920429e+02, 4.84004943e+02, 7.17241518e+02,
                    8.87482781e+02, 9.76466087e+02, 9.54130525e+02, 8.21581334e+02,
                    6.19085004e+02, 4.59888725e+02, 3.71295449e+02, 3.15689818e+02,
                    2.66624161e+02, 2.25634674e+02, 2.10376988e+02, 2.07897609e+02])
    d2 = np.array([170.6203983, 162.72957678, 150.83541175, 138.78669326, 122.24861362,
                    113.49124518, 91.34073072, 83.93452248, 106.68171974, 115.37107808,
                    123.3759738, 148.31675151, 249.29131364, 339.91796409, 463.67402264,
                    680.03072894, 891.37404325, 909.39742209, 720.29686162, 501.44087283,
                    298.15601013, 126.65120765, 27.76344419, 3.59881618, 51.63147899,
                    102.73432173, 122.70552028, 125.26718809, 131.90538957, 148.83504852,
                    157.45255021, 165.55309996])
    d1normalized, d2normalized = weighted_norm(d1, d2)

    with open(file, newline='') as csvfile:

        reader = csv.DictReader(csvfile)

        for row in reader:
            fit_values.append(row['fit_value'])
            parameters.append(ast.literal_eval(row['parameters']))
            func.append(row['function'])
            mirror.append(row['mirror'])
            dipole_force.append(row['dipole_force'])

    inc = np.deg2rad(parameters[row_num][0])
    pa = np.deg2rad(parameters[row_num][1])
    magco = np.deg2rad(parameters[row_num][2])
    phs = np.deg2rad(parameters[row_num][3])
    inc_off = np.deg2rad(parameters[row_num][4])
    az_off = np.deg2rad(parameters[row_num][5])
    r = parameters[row_num][6]
    gamma1 = parameters[row_num][7]
    w1 = parameters[row_num][8]
    w2 = parameters[row_num][9]
    gamma2 = parameters[row_num][10]
    phs2 = np.deg2rad(parameters[row_num][11])
    function = eval(func[row_num])

    dipole_force_now = dipole_force[row_num]
    if dipole_force_now == 'True':
        mirror_now = int(mirror[row_num])
    else:
        mirror_now = -1  # opposite pole-emission will be plotted to see what it would be

    intensities1 = get_intensity(inc, pa, magco, phs, inc_off, az_off, r, gamma1, function, w1, w2)
    intensities2 = get_intensity(inc, pa, (np.pi-magco), (np.pi+phs)%(2*np.pi)+phs2, mirror_now*inc_off,
                                 mirror_now*az_off, r, gamma1, function, w1, w2)

    # Plot pulse profile
    xarray = np.linspace(0, 1, 32)
    plt.figure()
    plt.plot(xarray, d1normalized, label="data", color = "blue", linestyle = "dashed")
    plt.plot(xarray, intensities1, label="best fit",  color = "blue")
    plt.plot(xarray, d2normalized, label="data" , color = "orange", linestyle = "dashed")
    plt.plot(xarray, intensities2, label="best fit", color = "orange")
    plt.title(f"Fit value = {round(float(fit_values[row_num]),3)} with " + criteria + " criteria")
    plt.legend()

    azimuths_adj, inclinations_adj, azimuths_adj2, inclinations_adj2 = spherePlot(inc, pa, magco, phs, r, phs2)
    az_off2 = mirror_now * az_off
    inc_off2 = mirror_now * inc_off
    emissionPatternPlot(azimuths_adj, inclinations_adj, azimuths_adj2, inclinations_adj2, az_off, inc_off, az_off2,
                        inc_off2, gamma1, gamma2, w1, w2, function)
    plt.show()

def spherePlot(inc, pa, magco, phs, r, phase_shift2):
    """Make sphere-plot."""
    magco2 = np.pi - magco
    phs2 = ((np.pi + phs) % (2 * np.pi)) + phase_shift2
    sphere_fig = plt.figure()
    sphereAx = sphere_fig.add_subplot(1, 1, 1, projection='3d')
    sphereAx.grid(visible=False)
    # sphereAx.set_axis_off()
    # sphereAx.set_zticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    sphereAx.set_xticks([-1.0, 0.0, 1.0])
    sphereAx.set_yticks([-1.0, 0.0, 1.0])
    sphereAx.set_zticks([-1.0, 0.0, 1.0])

    u = np.linspace(0, 2 * np.pi, 25)
    v = np.linspace(0, np.pi, 25)
    radius = 1
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    sphereAx.plot_surface(x, y, z, edgecolor='y', alpha=0.1, linewidth=0.15)  # sphere_plot
    v = [np.cos(inc) * np.cos(pa), np.cos(inc) * np.sin(pa), np.sin(inc)]
    v = v / np.linalg.norm(v)  # vector indicating spin axis

    u_0 = np.array([np.cos(inc + magco) * np.cos(pa),
                    np.cos(inc + magco) * np.sin(pa),
                    np.sin(inc + magco)])
    u_0 = u_0 / np.linalg.norm(u_0)
    u_0_2 = np.array([np.cos(inc + magco2) * np.cos(pa),
                      np.cos(inc + magco2) * np.sin(pa),
                      np.sin(inc + magco2)])
    u_0_2 = u_0_2 / np.linalg.norm(u_0_2)
    u_0 = rotate_vector(u_0, v, phs)
    u_0_2 = rotate_vector(u_0_2, v, phs2)
    sphereAx.quiver(0, 0, 0, v[0], v[1], v[2], color='k')  # spin_axis_plot
    sphereAx.quiver(0, 0, 0, u_0[0], u_0[1], u_0[2], color='purple')  # u_0_plot
    sphereAx.quiver(0, 0, 0, u_0_2[0], u_0_2[1], u_0_2[2], color='yellow')  # u_0_2_plot
    delta = np.pi / 16
    rotated_vectors = []
    rotated_vectors2 = []
    azimuths = [np.arctan2(u_0[1], u_0[0])]
    azimuths2 = [np.arctan2(u_0_2[1], u_0_2[0])]
    inclinations = [np.arcsin(u_0[2])]
    inclinations2 = [np.arcsin(u_0_2[2])]
    grav_angles = [np.arccos(u_0[0])]  # u dot direction(1, 0, 0)
    grav_angles2 = [np.arccos(u_0_2[0])]  # u dot direction(1, 0, 0)
    for j in range(31):
        rot_angle = delta * (j + 1)
        u_j = rotate_vector(u_0, v, rot_angle)
        u_j_2 = rotate_vector(u_0_2, v, rot_angle)
        rotated_vectors.append(u_j)
        rotated_vectors2.append(u_j_2)
        azimuths.append(np.arctan2(u_j[1], u_j[0]))
        azimuths2.append(np.arctan2(u_j_2[1], u_j_2[0]))
        inclinations.append(np.arcsin(u_j[2]))
        inclinations2.append(np.arcsin(u_j_2[2]))
        grav_angles.append(np.arccos(u_j[0]))
        grav_angles2.append(np.arccos(u_j_2[0]))
    azimuths_adj, inclinations_adj = gravity_adjust(azimuths, inclinations, grav_angles, radius=r)
    azimuths_adj2, inclinations_adj2 = gravity_adjust(azimuths2, inclinations2, grav_angles2, radius=r)
    rot_list = list(map(list, zip(*rotated_vectors)))  # Transpose the list to match testlist2 structure
    rot_list2 = list(map(list, zip(*rotated_vectors2)))  # Transpose the list to match testlist2 structure
    sphereAx.quiver(0, 0, 0, *rot_list, color='blue', alpha=0.2)  # u_j_plot
    sphereAx.quiver(0, 0, 0, *rot_list2, color='red', alpha=0.4)  # u_j_2_plot
    sphereAx.quiver(0, 0, 0, 1, 0, 0, color='green', alpha=1)  # direction_plot
    sphereAx.set_xlabel("X")
    sphereAx.set_ylabel("Y")
    sphereAx.set_zlabel("Z")
    sphereAx.set_box_aspect([1, 1, 1])
    return azimuths_adj, inclinations_adj, azimuths_adj2, inclinations_adj2


def emissionPatternPlot(azimuths_adj, inclinations_adj, azimuths_adj2, inclinations_adj2, az_off, inc_off, az_off2,
                        inc_off2, gamma1, gamma2, w1, w2, func):
    """Make emission pattern-plots."""
    feature_x = np.arange(-4, 4.05, 0.05)
    feature_y = np.arange(-4, 4.05, 0.05)
    [X, Y] = np.meshgrid(feature_x, feature_y)
    contour1fig = plt.figure()
    contour1Ax = contour1fig.add_subplot(1, 1, 1)
    contour2fig = plt.figure()
    contour2Ax = contour2fig.add_subplot(1, 1, 1)
    if func == combine:
        Z1 = combine(X, Y, gamma1, gamma2, az_off, inc_off, pencil, fan, w1, w2)
        Z2 = combine(X, Y, gamma1, gamma2, az_off2, inc_off2, pencil, fan, w1, w2)
    else:
        Z1 = func(X, Y, gamma1, az_off, inc_off)
        Z2 = func(X, Y, gamma2, az_off2, inc_off2)
    contour1Ax.contourf(X, Y, Z1)
    contour1Ax.scatter(azimuths_adj, inclinations_adj, linewidths=5, color='red')
    contour1Ax.scatter(azimuths_adj[0], inclinations_adj[0], linewidths=5, color='orange')
    contour1Ax.set_title("Main", fontsize=15)
    contour1Ax.set_xlim([-3.15, 3.15])
    contour1Ax.set_ylim([-1.6, 1.6])
    contour1Ax.set_xlabel('ϕ (rad)', fontsize=25)
    contour1Ax.set_ylabel('θ (rad)', fontsize=25)
    contour1Ax.tick_params(labelsize=20)

    contour2Ax.contourf(X, Y, Z2)
    contour2Ax.scatter(azimuths_adj2, inclinations_adj2, linewidths=5, color='pink')
    contour2Ax.scatter(azimuths_adj2[0], inclinations_adj2[0], linewidths=5, color='red')
    contour2Ax.set_title("Second", fontsize=15)
    contour2Ax.set_xlim([-3.15, 3.15])
    contour2Ax.set_ylim([-1.6, 1.6])
    contour2Ax.set_xlabel('ϕ (rad)', fontsize=25)
    contour2Ax.set_ylabel('θ (rad)', fontsize=25)
    contour2Ax.tick_params(labelsize=20)
    plt.show()
    
if __name__ == '__main__':
    criteria = "softmax" #file name without extension
    row_num = 100
    main(criteria, row_num-2)