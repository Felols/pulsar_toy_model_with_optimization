from sphere import Sphere, rotate_vector
from contour import Contour, pencil, fan, combine
from pulse_profile import PulseProfile
from matplotlib import pyplot as plt
from helper_functions import weighted_norm, get_intensity , gravity_adjust
import numpy as np
from numpy import pi
import csv
import timeit
import sys

def f_to_minimize(data1, data2, parameters, func, dipole_force, mirrored, phase_shifted_dipole, opti_type):
    """Judges how good fit given parameters are to data. Uses soft-max criteria.

    Args:
        data1 (_type_): _description_
        data2 (_type_): _description_
        parameters (list): list of parameters
        func (function): function for beam pattern
        dipole_force (bool, optional): describes if a forced dipole should be used
        mirrored (int): decribes if mirror or not
        opti_type (string) : details what should be returned


    Returns:
        float: fitting parameter
    """    
    inc = parameters[0]
    pa = parameters[1]
    magco = parameters[2]
    phs = parameters[3]
    inc_off = parameters[4]
    az_off = parameters[5]
    r = parameters[6]
    gamma = parameters[7]
    if phase_shifted_dipole:
        phs2 = parameters[10]
    else:
        phs2 = 0
    if func == combine:
        w1 = parameters[8]
        w2 = 1 - w1
        gamma2 = parameters[9]
        intensities1 = get_intensity(inc, pa, magco, phs, inc_off, az_off, r, gamma, func, w1, w2, pencil, fan, gamma2)
        intensities2 = get_intensity(inc, pa, (np.pi-magco), ((np.pi+phs+phs2)%(2*np.pi)), mirrored * inc_off,
                                     mirrored * az_off, r, gamma, func, w1, w2, pencil, fan, gamma2)
    else:
        intensities1 = get_intensity(inc, pa, magco, phs, inc_off, az_off, r , gamma, func)
        intensities2 = get_intensity(inc, pa, (np.pi-magco), ((np.pi+phs+phs2)%(2*np.pi)), mirrored * inc_off,
                                     mirrored * az_off, r, gamma, func)

    """intensities, data = weighted_norm(intensities, data)
    slopes = []
    slopes_data = []
    for j in range(1, len(data1)):
        slope = intensities[j] - intensities[j - 1]
        slope_data = data1[j] - data1[j - 1]
        slopes.append(slope)
        slopes_data.append(slope_data)
    errors = np.array(slopes_data) - np.array(slopes)"""

    errors1 = intensities1 - data1
    errors2 = intensities2 - data2
    s1 = 0
    s2 = 0
    for e in errors1:
        s1 += e ** 2
    for l in errors2:
        s2 += l ** 2
    if dipole_force:
        match opti_type:
            case "softmax":
                return np.log(np.exp(s1) + np.exp(s2))
            
            case "multi":
                return s1 * s2
            
            case "add":
                return s1 + s2
            
            case "harmonic":
                return 2 * (s1 * s2) / (s1 + s2)

            case default:
                sys.exit("Terminating program, faulty optimization type detected")
        #np.log(np.exp(s1) + np.exp(s2))  # Softmax-style combination
        #2 * (s1 * s2) / (s1 + s2)  # Harmonic mean
        #(s1 * s2) ** 0.5 Geometric mean
        #return np.log(np.exp(s1) + np.exp(s2))
    else:
        return s1


def runPSO(data1, data2, D, S, w, phi_p, phi_g, iterations, func, dipole_force, mirrored, phase_shifted_dipole,
           opti_type):
    """Find the best parameters to fit the data using Particle swarm optimization (PSO).

    Args:
        data1 (list): list of intensities for one pole
        data2 (list): list of intensities for second pole
        D (int): dimensions for PSO
        S (int): particles for PSO
        w (float): inertia weight for PSO
        phi_p (float): cognitive coefficient for PSO
        phi_g (float): social coefficient for PSO
        iterations (int): amount of iterations for PSO
        func (function): function to be used for beam pattern
        dipole_force (bool): describes if a forced dipole for should be used
        mirrored (int): describes if mirror or not
        phase_shifted_dipole (bool): describes if the second pole may be separately phase shifted
        opti_type (string) : details what should be returned from the f_to_minimize function

    Returns:
        list: the optimal parameters
        
    """    

    x = []  # particle positions. Contains: x = [inc, pa, magco, phs, inc_off, az_off, r, gamma, w1, gamma2, phs2]
    p = []  # particle best-known-positions
    v = []  # particle velocities
    g = []  # swarm best-known-position
    b_lo = np.array([-pi/2, -pi/2, 0, 0, -pi/4, -pi/4, 3, 1, 0.0, 1, 0])
    b_up = np.array([pi/2, pi/2, pi, 2*pi, pi/4, pi/4, 3.05, 5, 1.0, 5, 2*pi])
    diff = np.absolute(b_up - b_lo)
    for i in range(S):  # initialize
        x_i = np.random.uniform(b_lo, b_up)
        x.append(x_i.copy())
        p.append(x_i.copy())
        v_i = np.random.uniform(-diff, diff)
        v.append(v_i.copy())
        if i == 0:
            g = x_i.copy()
        else:
            if (f_to_minimize(data1, data2, p[i], func, dipole_force, mirrored, phase_shifted_dipole, opti_type)
                    < f_to_minimize(data1, data2, g, func, dipole_force, mirrored, phase_shifted_dipole, opti_type)):
                g = p[i].copy()

    print_stuff = True
    for n in range(iterations):
        distances = []
        failures = 0  # 1 failure is when a parameter tries going outside boundaries
        for i in range(S):
            for d in range(D):
                r_p = np.random.uniform(0, 1)
                r_g = np.random.uniform(0, 1)
                v[i][d] = w * v[i][d] + phi_p * r_p * (p[i][d] - x[i][d]) + phi_g * r_g * (g[d] - x[i][d])
                if b_up[d] >= x[i][d] + v[i][d] >= b_lo[d]:
                    x[i][d] = x[i][d] + v[i][d]
                else:  # move x[i][d] to boundary
                    failures += 1
                    if x[i][d] + v[i][d] > b_up[d]:
                        x[i][d] = b_up[d].copy()
                    elif x[i][d] + v[i][d] < b_lo[d]:
                        x[i][d] = b_lo[d].copy()
                    else:
                        print("error with boundary")
            if (f_to_minimize(data1, data2, x[i], func, dipole_force, mirrored, phase_shifted_dipole, opti_type)
                    < f_to_minimize(data1, data2, p[i], func, dipole_force, mirrored, phase_shifted_dipole, opti_type)):
                p[i] = x[i].copy()
                if (f_to_minimize(data1, data2, p[i], func, dipole_force, mirrored, phase_shifted_dipole, opti_type)
                        < f_to_minimize(data1, data2, g, func, dipole_force, mirrored, phase_shifted_dipole, opti_type)):
                    g = p[i].copy()
            diff = x[i] - g
            distance = np.dot(diff, diff)
            distances.append(distance)
        if print_stuff:
            print("max 'distance' to g:", np.max(distances))
            print("average 'distance' to g:", np.mean(distances))
            print("failures:", failures)
        print(n, g, f_to_minimize(data1, data2, g, func, dipole_force, mirrored, phase_shifted_dipole, opti_type))

    if not phase_shifted_dipole:
        g[10] = 0
    if func == combine:
        final_int_list1 = get_intensity(g[0], g[1], g[2], g[3], g[4], g[5], g[6], g[7], func, g[8], 1-g[8], pencil, fan,
                                        g[9])
        final_int_list2 = get_intensity(g[0], g[1], np.pi-g[2], (np.pi+g[3]+g[10])%(2*np.pi), mirrored*g[4],
                                        mirrored*g[5], g[6], g[7], func, g[8], 1-g[8], pencil, fan, g[9])
    else:
        final_int_list1 = get_intensity(g[0], g[1], g[2], g[3], g[4], g[5], g[6], g[7], func)
        final_int_list2 = get_intensity(g[0], g[1], np.pi-g[2], (np.pi+g[3]+g[10])%(2*np.pi), mirrored*g[4],
                                        mirrored*g[5], g[6], g[7], func)
    if dipole_force:
        return g, final_int_list1, final_int_list2
    else:
        return g, final_int_list1, None


def PSO(data1, data2):
    """Calls the PSO, then plots the result and stores them in the corresponding to opti_type csv file.

    Args:
        data1 (list): data to fit to
        data2 (list): data to fit to
    """

    ## PSO variables and scenarios
    func = pencil # Funcs are pencil, fan, combine
    dipole_force = True
    mirrored = -1   # 1 for no mirror, -1 for mirror
    phase_shifted_dipole = False
    record_time = True  # False to not include runtime, True to include runtime
    opti_type = "softmax" # Types are "softmax", "multi", "add", "harmonic"
    S = 50  # particles
    D = 11  # dimensions
    w = 0.4  # inertia weight
    phi_p = 1  # cognitive coefficient
    phi_g = 2  # social coefficient
    iterations = 50
    ##

    start_time = timeit.default_timer()
    parameters, intensities1, intensities2 = runPSO(data1, data2, D, S, w, phi_p, phi_g, iterations, func,
                                                     dipole_force, mirrored, phase_shifted_dipole, opti_type)
    if record_time:
        run_time = timeit.default_timer()-start_time
    else:
        run_time = None
    fit_value = f_to_minimize(data1, data2, parameters, func, dipole_force, mirrored, phase_shifted_dipole, opti_type)
    print(fit_value)

    # add w2 to parameters-list before saving data
    g_final = [parameters[i] for i in range(9)]  # g-values up until including w1
    g_final.append(1 - parameters[8])  # add w2 at index 9
    g_final.append(parameters[9])  # add gamma2 at index 10
    g_final.append(parameters[10])  # add phs2 at index 11
    parameters = g_final
    xarray = np.linspace(0, 1, 32)
    fig = plt.figure()
    plt.plot(xarray, data1, label="data", color = "blue", linestyle = "dashed")
    plt.plot(xarray, intensities1, label="best fit",  color = "blue")
    if dipole_force:
        plt.plot(xarray, data2, label="data" , color = "orange", linestyle = "dashed")
        plt.plot(xarray, intensities2, label="best fit", color = "orange")
    else:
        pass
        #plt.plot(xarray, data2, label="data" , color = "red", linestyle = "dashed")
        #plt.plot(xarray, intensities2, label="best fit", color = "red")
    plt.title(func.__name__)
    plt.legend()

    parameters_degrees = []
    for i in range(len(parameters)):
        if i <= 5 or i==11:
            parameters_degrees.append(np.rad2deg(parameters[i]))
        else:
            parameters_degrees.append(parameters[i])
    rounded_list = [round(num, 3) for num in parameters_degrees]

    csvwrite = [parameters, fit_value, mirrored, func, dipole_force, S, iterations, w, phi_p, phi_g]

    with open(opti_type + '.csv' , 'a', newline='') as csvfile:  # a means append mode
        fieldnames = ['parameters', 'fit_value', "mirror", "function", "dipole_force",
                      "Particles", "iterations", "Inertia", "Cognitive", "Social", "Runtime"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerows([{"parameters" : rounded_list,
                            "fit_value" : fit_value,
                            "mirror" : mirrored,
                            "function" : func.__name__ ,
                            "dipole_force" : dipole_force,
                            "Particles" : S ,
                            "iterations" : iterations,
                            "Inertia" : w,
                            "Cognitive" : phi_p ,
                            "Social" : phi_g,
                            "Runtime" : run_time}])
    plt.show()


def plot_from_values(inc, pa, magco, phs, inc_off, az_off, r, gamma, func=pencil, n=32):
    """Plots the phase profile for some set values
    Args:
        inc (float): inclination
        pa (float): positional angle
        magco (float): magnetic offset also called magnetic colatitude
        phs (float): phase shift
        inc_off (float): beam pattern offset for inclination
        az_off (float): beam pattern offset for azimuth
        r (float): radius in Schwarzschild ratio
        gamma (float): exponent
        func (function, optional): function for beam pattern. Defaults to f.
        n (int, optional): amount of data points. Defaults to 32.
    """    
    intensity = get_intensity(inc, pa, magco, phs, inc_off, az_off, r, gamma, func, n)
    xarray = np.linspace(0, 1, 32)
    fig = plt.figure()
    plt.plot(xarray, intensity, label="best fit",  color = "blue")
    plt.show()


def main():
    """Runs the PSO.
    """   
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

    PSO(d1normalized, d2normalized)

if __name__ == '__main__':
    main()