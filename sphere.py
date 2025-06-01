import numpy as np
import matplotlib.pyplot as plt
from contour import pencil
from helper_functions import rotate_vector, gravity_adjust

class Sphere:
    """Defines a pulsar to be plotted, including a sphere surface and different vectors."""
    def __init__(self, fig, nrows, ncols, index):
        """Initializes future references to plots and lists of values.

        Args:
            fig (figure): matplotlib figure
            nrows (int): determines where the figure is placed
            ncols (int): determines where the figure is placed
            index (int): determines where the figure is placed
        """
        self.fig = fig
        self.ax = self.fig.add_subplot(nrows, ncols, index, projection='3d')

        self.sphere_plot = None
        self.spin_axis_plot = None
        self.u_0_plot = None
        self.u_j_plot = None
        self.azimuths = None
        self.azimuths_adj = None
        self.inclinations = None
        self.inclinations_adj = None
        self.grav_angles = None
        self.int = None
        self.int_adj = None

        self.i_init = 0
        self.offset_init = 16
        self.phase_shift_init = 0
        self.radius_init = 3

        self.data_points = 32

    def plot_sphere(self):
        """Plots the outline of a sphere."""
        # Define u and v angles
        u = np.linspace(0, 2 * np.pi, 25)
        v = np.linspace(0, np.pi, 25)
        radius = 1
        # Parametric equations for the sphere
        x = radius * np.outer(np.cos(u), np.sin(v))
        y = radius * np.outer(np.sin(u), np.sin(v))
        z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
        self.sphere_plot = self.ax.plot_surface(x, y, z, edgecolor='y', alpha=0.1, linewidth = 0.15)

    def plot_vectors(self, inc=0, pa=0, magco=16, phs=0, n=32):
        """Plots spin axis vector and vectors for the positions of an emission point.

        Args:
            inc (float, optional): inclination of the rotation axis (degrees). Defaults to 0.
            pa (float, optional): positional angle of the rotation axis (degrees). Defaults to 0.
            magco (float, optional): magnetic colatitude (degrees). Defaults to 16.
            phs (float, optional): phase shift (degrees). Defaults to 0.
            n (int, optional): amount of data points. Defaults to 32.
        """        
        inc = np.deg2rad(inc)
        pa = np.deg2rad(pa)
        magco = np.deg2rad(magco)
        phs = np.deg2rad(phs)
        v = [np.cos(inc) * np.cos(pa), np.cos(inc) * np.sin(pa), np.sin(inc)]
        v = v / np.linalg.norm(v)  # vector indicating spin axis
        u_0 = np.array([np.cos(inc + magco) * np.cos(pa),
                      np.cos(inc + magco) * np.sin(pa),
                      np.sin(inc + magco)])
        u_0 = u_0 / np.linalg.norm(u_0)
        u_0 = rotate_vector(u_0, v, phs)
        
        self.spin_axis_plot = self.ax.quiver(0, 0, 0, v[0], v[1], v[2], color='k')
        self.u_0_plot = self.ax.quiver(0, 0, 0, u_0[0], u_0[1], u_0[2], color='purple')

        delta_phi = np.pi / (n/2)
        rotated_vectors = []
        self.azimuths = [np.arctan2(u_0[1], u_0[0])]
        self.inclinations = [np.arcsin(u_0[2])] # should be divided by r but r is defined to be 1 for all vectors
        self.grav_angles = [np.arccos(u_0[0])] # vector dot direction (1, 0, 0) divided by their lengths (both are 1)
        for j in range(n-1):
            u_j = rotate_vector(u_0, v, delta_phi*(j+1))
            rotated_vectors.append(u_j)
            self.azimuths.append(np.arctan2(u_j[1], u_j[0]))
            self.inclinations.append(np.arcsin(u_j[2]))
            self.grav_angles.append(np.arccos(u_j[0]))
        self.azimuths_adj, self.inclinations_adj = gravity_adjust(self.azimuths, self.inclinations, self.grav_angles,
                                                                  radius=3)
        self.int = pencil(self.azimuths, self.inclinations, 2, 0, 0)
        self.int_adj = pencil(self.azimuths_adj, self.inclinations_adj, 2, 0, 0)

        rot_list = list(map(list, zip(*rotated_vectors)))  # Transpose rotated_vectors to match testlist2 structure
        self.u_j_plot = self.ax.quiver(0, 0, 0, *rot_list, color='blue', alpha=0.4)

        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_title("Pulsar Geometry", fontsize = 15)
        self.ax.set_box_aspect([1, 1, 1])


def main():
    """Testing function of this file."""
    fig = plt.figure(figsize=(10, 7))
    sphere = Sphere(fig, 2, 3, 1)
    sphere.plot_sphere()
    sphere.plot_vectors(10, -60, 20)
    plt.show()


if __name__ == '__main__':
    main()