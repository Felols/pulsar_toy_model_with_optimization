from contour import pencil, fan, combine
import numpy as np


def rotate_vector(vector, axis, angle):
    """A function for rotating a vector around an axis with a specific angle in radians
    Uses Rodrigues rotation formula

    Args:
        vector (list): the vector to rotate
        axis (list): the axis to rotate around
        angle (float): the angle to rotate with in radians

    Returns:
        list: rotate vector
    """    
    # Using Rodrigues rotation formula
    vector = np.array(vector)
    axis = np.array(axis)
    axis = axis / np.linalg.norm(axis)
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    a = vector * cos_theta
    b = np.cross(axis, vector) * sin_theta
    c = axis * np.dot(axis, vector) * (1 - cos_theta)
    return a + b + c


def weighted_norm(list1, list2):
    """Weights two lists such that their sum is normalized.

    Args:
        list1 (list): data list 1
        list2 (list): data list 2

    Returns:
        list: normalized list
    """    
    list_sum = list1 + list2
    max_value = max(list_sum)
    normedlist_1 = list1 / max_value
    normedlist_2 = list2 / max_value

    return normedlist_1, normedlist_2


def get_intensity(inc, pa, magco, phs, inc_off, az_off, r, gamma, func, w1=0.7, w2=0.3, func1 = pencil,
                  func2=fan, gamma2=2, n=32):
    """Calculates list of intensities (adjusted for light bending) for given parameters.

    Args:
        inc (float): inclination
        pa (float): positional angle
        magco (float): magnetic offset also called magnetic colatitude
        phs (float): phase shift
        inc_off (float): beam pattern offset for inclination
        az_off (float): beam pattern offset for azimuth
        r (float, optional): radius in Schwarzschild
        gamma (float, optional): exponent
        func (function, optional): function for beam pattern
        w1 (float, optional): used if func is combine, then determines weight of func1. Defaults to 0.7.
        w2 (float, optional): used if func is combine, then determines weight of func2. Defaults to 0.3.
        func1 (function, optional): to be used if func is combine. Defaults to pencil.
        func2 (function, optional): used if func is combine. Defaults to fan.
        n (int, optional): amount of data points. Defaults to 32.
        gamma2 (float, optional): exponent for the second function in combine. Defaults to 2.

    Returns:
        list: intensities
    """    
    v = [np.cos(inc) * np.cos(pa), np.cos(inc) * np.sin(pa), np.sin(inc)]
    v = v / np.linalg.norm(v)  # vector indicating spin axis
    u_0 = np.array([np.cos(inc + magco) * np.cos(pa), np.cos(inc + magco) * np.sin(pa), np.sin(inc + magco)])
    u_0 = u_0 / np.linalg.norm(u_0)
    u_0 = rotate_vector(u_0, v, phs)
    azimuths = [np.arctan2(u_0[1], u_0[0])]
    inclinations = [np.arcsin(u_0[2])]  # should be divided by r but r is defined to be 1 for all vectors
    grav_angles = [np.arccos(u_0[0])]  # vector dot direction (1, 0, 0) divided by their lengths (which both are 1)

    delta = np.pi / (n / 2)
    for j in range(n - 1):
        rot_angle = delta * (j + 1)
        u_j = rotate_vector(u_0, v, rot_angle)
        azimuths.append(np.arctan2(u_j[1], u_j[0]))
        inclinations.append(np.arcsin(u_j[2]))
        grav_angles.append(np.arccos(u_j[0]))
    # int_list = func(azimuths, inclinations, gamma, i_offset, az_offset)
    azimuths_adj, inclinations_adj = gravity_adjust(azimuths, inclinations, grav_angles, radius=r)
    if func == combine:
        int_adjusted = func(azimuths_adj, inclinations_adj, gamma, gamma2, az_off, inc_off, func1, func2, w1, w2)
    else:
        int_adjusted = func(azimuths_adj, inclinations_adj, gamma, az_off, inc_off)
    return int_adjusted


def rotate_vector_grav(vector, axis, angle):
    """A modified version of rotate vector for use in the gravity adjust function.

    Args:
        vector (list): the vector to rotate
        axis (list): the axis to rotate around
        angle (float): the angle to rotate with in radians

    Returns:
        list: rotated vector
    """    
    vector = np.array(vector)
    axis = np.array(axis)
    axis = axis / np.linalg.norm(axis)
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    a = vector * cos_theta
    b = np.cross(axis, vector) * sin_theta
    #c = axis * np.dot(axis, vector) * (1 - cos_theta)
    c = 0   #Because by construction np.dot(axis, vector) is always zero (they are orthogonal) this saves some computations
    return a + b + c

def gravity_adjust(azimuths, inclinations, psi_angles, radius):
    """Adjusts the vector defined by the inputs with a Beloborodov gravity formula dependent on radius.

    Args:
        azimuths (list): collection of azimuths of pole position
        inclinations (list): collection of inclinations of pole positions
        psi_angles (list): collection of angle between pole and observer vector
        radius (float): radius of pulsar, in units of Schwarzschild radius (r_G)

    Returns:
        list, list: collection of adjusted azimuth and inclinations
    """
    # limits of model at R > 2r_g
    # 1 - cos(alpha) = (1 - cos(psi) * (1 - r_G/R)
    # -> cos(alpha) = 1 - ((1 - cos(psi)) * (1 - r_g/R))
    # where alpha is the bending angle, r_G is Schwarzschild radius, R is radius of pulsar.
    
    Belo_righthand = (1 - np.cos(psi_angles)) * (1-(1/radius))
    cos_expression = 1 - Belo_righthand
    beta_angles = np.arccos(cos_expression)
    bending_angles = psi_angles - beta_angles
    
    #Reconstructs vectors
    x = np.cos(inclinations) * np.cos(azimuths)
    y = np.cos(inclinations) * np.sin(azimuths)
    z = np.sin(inclinations)
    
    reconstructed_vectors = list(zip(x,y,z))

    obs_direction = [1, 0, 0]
    ortho_vectors = np.cross(reconstructed_vectors, obs_direction)

    azimuths_adjusted = np.full_like(azimuths, 1)
    inclinations_adjusted = np.full_like(azimuths, 1)
    
    for j in range(len(azimuths)):
            u_j = rotate_vector_grav(reconstructed_vectors[j], ortho_vectors[j], bending_angles[j])
            azimuths_adjusted[j] = (np.arctan2(u_j[1], u_j[0]))
            inclinations_adjusted[j] = (np.arcsin(u_j[2]))

    return azimuths_adjusted, inclinations_adjusted
