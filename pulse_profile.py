from matplotlib import pyplot as plt
from sphere import Sphere, rotate_vector
from contour import Contour, pencil, fan, combine
import numpy as np
#from scipy import interpolate


class PulseProfile:
    def __init__(self, fig, nrows, ncols, index):
        """Initializes a pulse profile object with specified figure and subplot configuration.

        Args:
            fig (figure): matplotlib figure
            nrows (int): determines where the figure is placed
            ncols (int): determines where the figure is placed
            index (int): determines where the figure is placed
        """
        self.fig = fig
        self.ax = self.fig.add_subplot(nrows, ncols, index)
        self.int_plot = None
        self.int_adj_plot = None

    def pulse_plot(self, intensities, intensities_adj, data_points):
        """Plots the pulse profile including normal and gravity-adjusted intensities, normalized to a phase.

        Args:
            intensities (list): List of intensities
            intensities_adj (list): List of adjusted intensities
            data_points (int): Amount of data points for one phase
        """        
        self.ax.set_ylim(-0.05, 1.1)
        self.ax.set_xlabel("Phase")
        self.ax.set_ylabel("Relative Intensity")
        self.ax.set_title("Pulse Profile", fontsize = 15)
        self.int_plot = self.ax.plot(np.linspace(0,1,data_points), intensities, color = "red")
        self.int_adj_plot = self.ax.plot(np.linspace(0,1,data_points), intensities_adj, color = "orange")


def main():
    """Testing function for this python file."""
    fig = plt.figure(figsize=(7, 7))
    cont = Contour(fig, 2, 3, 2, pencil)
    sphere = Sphere(fig, 2, 3, 1)
    test_int = [np.sin(x/(16*np.pi)) for x in range(32)]
    test_int_adj = [np.sin(x/(8*np.pi)) for x in range(32)]
    pulseprof = PulseProfile(fig, 1, 1, 1)
    PulseProfile.pulse_plot(test_int, test_int_adj, 32)
    plt.show()


if __name__ == '__main__':
    main()