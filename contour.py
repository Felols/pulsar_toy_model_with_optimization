import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm


def pencil(x, y, gamma, x_off, y_off):
    """A beam pattern function for pencil beams.

    Args:
        x (list): list of azimuth
        y (list): list of inclination
        gamma (float): the exponent
        x_off (float): azimuth offset (radians)
        y_off (float): inclination offset (radians)

    Returns:
        float: relative intensity
    """    
    x = [i+x_off for i in x]
    y = [j+y_off for j in y]
    return (np.cos(x) * np.cos(y) * np.heaviside(np.cos(x),0) * np.heaviside(np.cos(y),0)) ** gamma

def fan3(x, y, gamma , x_off, y_off):
    """A beam pattern function for a simple fan-beam like pattern.

    Args:
        x (list): list of azimuth
        y (list): list of inclination
        gamma (float): the exponent
        x_off (float): azimuth offset (radians)
        y_off (float): inclination offset (radians)

    Returns:
        float: relative intensity
    """  
    fval = pencil(x, y, gamma , x_off, y_off)
    return (1 - fval) * np.heaviside(np.cos(x),0) * np.heaviside(np.cos(y),0)

def fan(x, y, gamma, x_off, y_off):
    """A beam pattern function for a fan-beam like pattern.

    Args:
        x (list): list of azimuth
        y (list): list of inclination
        gamma (float): the exponent
        x_off (float): azimuth offset (radians)
        y_off (float): inclination offset (radians)

    Returns:
        float: relative intensity
    """  
    beta = np.arccos(np.cos(y + y_off) * np.cos(x + x_off))
    BETA = np.abs(beta)
    return (BETA / (np.pi/2))**gamma * (np.heaviside(BETA, 0) - np.heaviside(BETA-np.pi/2, 0)) + (1 - (10/np.pi) *
            (BETA - np.pi/2)) * (np.heaviside(BETA-np.pi/2, 0) - np.heaviside(BETA-6*np.pi/10, 0))

def fan2(x, y, gamma , x_off, y_off):
    """A beam pattern function for a fan-beam like pattern.

    Args:
        x (list): list of azimuth
        y (list): list of inclination
        gamma (float): the exponent
        x_off (float): azimuth offset
        y_off (float): inclination offset

    Returns:
        float: relative intensity
    """  
    r = np.sqrt(np.square(x + x_off)+np.square(y + y_off))
    return (r / (np.pi / 2))**gamma * (np.heaviside(r, 0) - np.heaviside(r - np.pi / 2, 0)) + (1 - (10 / np.pi) *
            (r - np.pi / 2)) * (np.heaviside(r - np.pi / 2, 0) - np.heaviside(r - 6 * np.pi / 10, 0))

def pencil2(x,y, gamma , x_off, y_off):
    """A beam pattern function for a pencil beam like pattern using gaussians.

    Args:
        x (list): list of azimuth
        y (list): list of inclination
        gamma (float): the exponent, unused
        x_off (float): azimuth offset
        y_off (float): inclination offset

    Returns:
        float: relative intensity
    """  
    sigma = np.pi/8
    x = np.array(x + x_off)
    y = np.array(y + y_off)
    max_phi = np.pi/2
    return np.exp((-x ** 2) / (2 * (sigma ** 2))) * np.exp((-y ** 2) / (2 * (sigma ** 2)))

def combine(x,y, gamma1, gamma2 , x_off, y_off, func1, func2, w1, w2):
    """A beam pattern function for a combined pencil and fan beam like pattern using a weighted sum.

        Args:
            x (list): list of azimuth
            y (list): list of inclination
            gamma1 (float): the exponent for func1
            gamma2 (float): the exponent for func2
            x_off (float): azimuth offset
            y_off (float): inclination offset
            func1 (function): an emission pattern-function
            func2 (function): an emission pattern-function
            w1 (float): weight for func1 in the sum, should equal 1-w2
            w2 (float): weight for func2 in the sum, should equal 1-w1

        Returns:
            float: relative intensity
    """
    feature_x = np.arange(-3.15, 3.2, 0.05)
    feature_y = np.arange(-3.15, 3.2, 0.05)
    [X, Y] = np.meshgrid(feature_x, feature_y)
    Z1 = func1(X, Y, gamma1, x_off, y_off)
    Z2 = func2(X, Y, gamma2, x_off, y_off)
    max_value = np.max(w1 * Z1 + w2* Z2)
    return (w1*func1(x,y, gamma1 , x_off, y_off) + w2*func2(x,y, gamma2 , x_off, y_off))/max_value


class Contour:
    """Encloses the relevant parameters for a contour plot, having the properties of X and Y being grids for position
    and Z being a grid of values that a function f takes at every discrete position."""
    def __init__(self, fig, nrows, ncols, index, func = pencil):
        """Initializes a contour with specified figure, subplot configuration and emission pattern-function.

        Args:
            fig (figure): matplotlib figure
            nrows (int): determines where the figure is placed
            ncols (int): determines where the figure is placed
            index (int): determines where the figure is placed
            func (function, optional): function to contour plot. Defaults to f.
        """        
        # Creating 2-D grid of features
        feature_x = np.arange(-4, 4.05, 0.05)
        feature_y = np.arange(-4, 4.05, 0.05)
        [X, Y] = np.meshgrid(feature_x, feature_y)
        self.X = X
        self.Y = Y
        self.Z = None
        self.ax = fig.add_subplot(nrows, ncols, index, xmargin = 15, box_aspect = 0.5)
        self.func = func

    def contour_plot(self, azimuths, inclinations, azimuths_adj, inclinations_adj, az_offset=0, inc_offset=0, gamma1=2,
                     plot_coordinates=True, gamma2=2, w1=0.5, w2=0.5):
        """Creates a contour plot in a subplot placed in the figure fig, which should contain a grid of subplots where
        this contour-plot can be placed at 'index'.

        Args:
            azimuths (list): list of azimuth values before adjustment
            inclinations (list): list of inclination values before adjustment
            azimuths_adj (list): list of azimuth values after adjustment
            inclinations_adj (list): list of inclination values after adjustment
            az_offset (float): Specifies how much the center is offset in the azimuthal direction. Default is 0.
            inc_offset (float): Specifies how much the center is offset in the inclinational direction. Default is 0.
            gamma1 (float): Larger gamma makes intensity more concentrated toward the maximum location(s). Default is 2.
            plot_coordinates (bool): Impacts how the emission pattern-plot is made. Should only be False when making
             emission pattern-plots with contour.main(). Default is True.
            gamma2 (float): Works as gamma1, but for the fan beam when combined beam is used. Default is 2.
            w1 (float): Between 0 and 1. Specifies the percentage of pencil-beam when combine-beam is used.
             Default is 0.5.
            w2 (float): Between 0 and 1. Should equal 1-w1, specifies the percentage of fan-beam when combine beam is
             used. Default is 0.5.
        """
        if self.func == combine:
            self.Z = combine(self.X, self.Y, gamma1, gamma2, az_offset, inc_offset, pencil, fan, w1, w2)
        else:
            self.Z = self.func(self.X, self.Y, gamma1, az_offset, inc_offset)
        self.ax.contourf(self.X, self.Y, self.Z)
        if plot_coordinates:
            self.ax.scatter(azimuths, inclinations, linewidths=0.5, color='red')
            self.ax.scatter(azimuths[0], inclinations[0], linewidths=0.5, color='yellow')
            self.ax.scatter(azimuths_adj, inclinations_adj, linewidths=0.5, color='orange')
            self.ax.scatter(azimuths_adj[0], inclinations_adj[0], linewidths=0.5, color='white')
            self.ax.set_title('Emission Pattern', fontsize=15)
            self.ax.set_xlim([-3.15, 3.15])
            self.ax.set_ylim([-1.6, 1.6])
            self.ax.set_xlabel('ϕ (rad)', fontsize=10)
            self.ax.set_ylabel('θ (rad)', fontsize=10)
            self.ax.tick_params(labelsize=10)
            self.ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
            self.ax.set_yticks([-1.5, 0, 1.5])
        else:
            cbar = plt.colorbar(self.ax.contourf(self.X, self.Y, self.Z), None, self.ax, ticks=[0, 1], pad=0.01)
            cbar.ax.tick_params(labelsize=30)
            cbar.set_label("Rel. Int.", fontsize=30)
            # self.ax.set_title('Emission Pattern', fontsize=15)
            self.ax.set_xlim([-3.15, 3.15])
            self.ax.set_ylim([-1.6, 1.6])
            self.ax.set_xlabel('ϕ (rad)', fontsize=30)
            self.ax.set_ylabel('θ (rad)', fontsize=30)
            self.ax.tick_params(labelsize=30)
            plt.tight_layout()


def plotBetaFunc(gamma, color):
    betas = np.arange(-2.50, 2.51, 0.01)
    funcvals = (betas / (np.pi / 2)) ** gamma * (np.heaviside(betas, 0) - np.heaviside(betas - np.pi / 2, 0)) + (
                1 - (10 / np.pi)
                * (betas - np.pi / 2)) * (
                       np.heaviside(betas - np.pi / 2, 0) - np.heaviside(betas - 6 * np.pi / 10, 0))
    plt.plot(betas, funcvals, label="γ=" + str(gamma), color=color)
    plt.xlabel("β", fontsize=15)
    plt.ylabel("Rel. Int.", fontsize=15)


def multipleBetaFuncs():
    colors = cm.Blues(np.linspace(0.4, 1, 6))
    plotBetaFunc(1, colors[0])
    plotBetaFunc(2, colors[1])
    plotBetaFunc(3, colors[2])
    plotBetaFunc(4, colors[3])
    plotBetaFunc(5, colors[4])
    plotBetaFunc(10, colors[5])
    plt.legend(loc=2)
    plt.show()


def main():
    """Main/testing function for this python file."""
    contour_fig = plt.figure()
    contour = Contour(contour_fig, 1, 1, 1, fan)
    contour.contour_plot(None, None, None, None, 0, 0, 1, False)

    """contour2_fig = plt.figure()
    contour2 = Contour(contour2_fig, 1, 1, 1, h1)
    contour2.contour_plot(None, None, None, None, 0, 0, 5, False)

    contour3_fig = plt.figure()
    contour3 = Contour(contour3_fig, 1, 1, 1, h1)
    contour3.contour_plot(None, None, None, None, np.pi/6, 0, 2, False)

    contour4_fig = plt.figure()
    contour4 = Contour(contour4_fig, 1, 1, 1, h1)
    contour4.contour_plot(None, None, None, None, 0, np.pi/2, 2, False)"""
    plt.show()
    # multipleBetaFuncs()


if __name__ == '__main__':
    main()