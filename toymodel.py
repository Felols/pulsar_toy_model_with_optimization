from sliders import Sliders
from sphere import Sphere
from contour import Contour, pencil, fan, combine
from pulse_profile import PulseProfile
from matplotlib import pyplot as plt
import numpy as np
from numpy import pi



def run():
    """Runs the program using a Sphere and a Contour plot"""
    fig = plt.figure(figsize=(15, 7))
    sphere = Sphere(fig, 2, 3, 1)
    sphere.plot_sphere()
    sphere.plot_vectors()
    contour = Contour(fig, 2, 3, 2)
    contour.contour_plot(sphere.azimuths, sphere.inclinations, sphere.azimuths_adj, sphere.inclinations_adj)
    pulse_prof = PulseProfile(fig, 2, 3, 3)
    sliders = Sliders(fig, sphere, contour, pulse_prof,)
    sliders.plot_sliders()
    pulse_prof.pulse_plot(sphere.int, sphere.int_adj, sphere.data_points)
    plt.subplots_adjust(wspace=0.25)
    # fig.canvas.flush_events()
    # fig.tight_layout()
    plt.show()



def main():
    """Runs the toy model.
    """   
    run()
    

if __name__ == '__main__':
    main()