from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, TextBox, RadioButtons, Button
from pulse_profile import PulseProfile
from sphere import Sphere
from contour import Contour, pencil, fan, combine
from helper_functions import gravity_adjust
import numpy as np


class Sliders:
    def __init__(self, fig, sphere, contour, pulse_prof):
        """Initializes Sliders properties."""
        self.fig = fig
        self.inc_slider = None
        self.pa_slider = None
        self.magco_slider = None
        self.phs_slider = None
        self.r_slider = None
        self.az_off_slider = None
        self.inc_off_slider = None
        self.gamma1_slider = None
        self.gamma2_slider = None
        self.w1_slider = None
        self.w2_textbox = None

        self.gamma2_slider_ax = None
        self.w1_slider_ax = None
        self.w2_textbox_ax = None

        self.text_box = None
        self.int_plot = None

        self.sphere = sphere
        self.contour = contour
        self.pulse_prof = pulse_prof
        self.func = pencil
        self.function_map = {'pencil': pencil, 'fan': fan, 'combine': combine}
        self.func_radios = None
        
    """def box_change(self, val):
        new_val = float(val)
        if -90 <= new_val <= 90:
            self.inc_slider.set_val(new_val)
        else:
            self.text_box.set_val(f"{self.pa_slider.val:.2f}")"""

    """def w2_box_change(self, val):
        if val<0:
            self.w2_textbox.set_val(0)
        elif val>1:
            self.w2_textbox.set_val(1)
        self.w1_slider.set_val(1-self.w2_textbox.val)"""


    def sphere_slider_on_changed(self, val):
        """Updates the sphere-plot, emission pattern and pulse profile when certain parameter-sliders have been
        changed."""
        self.sphere.ax.clear()
        self.pulse_prof.ax.clear()
        new_inc = self.inc_slider.val
        new_pa = self.pa_slider.val
        new_offset = self.magco_slider.val
        new_phase_shift = self.phs_slider.val
        self.sphere.plot_sphere()
        self.sphere.plot_vectors(new_inc, new_pa, new_offset, new_phase_shift)
        #self.int_plot = self.phase_prof.ax.plot(self.sphere.int)
        #self.text_box.set_val(round(self.i_slider.val, 2))
        self.contour_slider_on_changed(val)

    def contour_slider_on_changed(self, val):
        """Updates the emission pattern and the pulse profile when certain parameter-sliders have been changed."""
        self.pulse_prof.ax.clear()
        radius = self.r_slider.val
        az_off = self.az_off_slider.val
        inc_off = self.inc_off_slider.val
        gamma1 = self.gamma1_slider.val
        self.sphere.azimuth_adj, self.sphere.inclinations_adj = gravity_adjust(self.sphere.azimuths,
                                    self.sphere.inclinations, self.sphere.grav_angles, radius)
        if self.func == combine:
            gamma2 = self.gamma2_slider.val
            w1 = self.w1_slider.val
            w2 = 1-w1
            self.w2_textbox.set_val(round(w2,3))
            self.sphere.int_adjusted = combine(self.sphere.azimuth_adj, self.sphere.inclinations_adj, gamma1, gamma2,
                                               np.deg2rad(az_off), np.deg2rad(inc_off), pencil, fan, w1, w2)
            self.sphere.int = combine(self.sphere.azimuths, self.sphere.inclinations, gamma1, gamma2,
                                      np.deg2rad(az_off), np.deg2rad(inc_off), pencil, fan, w1, w2)
            self.contour.contour_plot(self.sphere.azimuths, self.sphere.inclinations,
                                      self.sphere.azimuths_adj, self.sphere.inclinations_adj, np.deg2rad(az_off),
                                      np.deg2rad(inc_off), gamma1, True, gamma2, w1, w2)
        else:
            self.sphere.int_adj = self.contour.func(self.sphere.azimuths_adj, self.sphere.inclinations_adj,
                                                         gamma1, np.deg2rad(az_off), np.deg2rad(inc_off))
            self.sphere.int = self.contour.func(self.sphere.azimuths, self.sphere.inclinations, gamma1,
                                                np.deg2rad(az_off), np.deg2rad(inc_off))
            self.contour.contour_plot(self.sphere.azimuths, self.sphere.inclinations,
                                      self.sphere.azimuths_adj, self.sphere.inclinations_adj, np.deg2rad(az_off),
                                      np.deg2rad(inc_off), gamma1)
        self.int_plot = self.pulse_prof.pulse_plot(self.sphere.int, self.sphere.int_adj,
                                                   self.sphere.data_points)
        self.fig.canvas.draw_idle()
        
    
    def func_radios_on_changed(self, val):
        """Defines actions of program when a new function is chosen."""
        self.contour.func = self.function_map.get(val, None)
        self.func = self.function_map.get(val, None)
        if self.func == combine:
            self.gamma2_slider_ax.set_visible(True)
            self.w1_slider_ax.set_visible(True)
            self.w2_textbox_ax.set_visible(True)
        else:
            self.gamma2_slider_ax.set_visible(False)
            self.w1_slider_ax.set_visible(False)
            self.w2_textbox_ax.set_visible(False)
        self.contour_slider_on_changed(val)

    def plot_sliders(self, inc_init=0, pa_init=0, magco_init=16, phs_init=0, r_init=3, az_off_init=0, inc_off_init=0,
                     gamma1_init=2, gamma2_init=2, w1_init=0.5, w2_init=0.5):
        """Defines slider for different parameters."""
        axis_color = 'lightgoldenrodyellow'
        axis_color2 = 'purple'
        inc_slider_ax = self.fig.add_axes([0.25, 0.10, 0.55, 0.03], facecolor=axis_color)
        self.inc_slider = Slider(inc_slider_ax, 'Inclination', -90, 90, valinit=inc_init, dragging=True)
        pa_slider_ax = self.fig.add_axes([0.25, 0.15, 0.55, 0.03], facecolor=axis_color)
        self.pa_slider = Slider(pa_slider_ax, 'Positional angle', -90, 90, valinit=pa_init, dragging=True)
        magco_slider_ax = self.fig.add_axes([0.25, 0.05, 0.55, 0.03], facecolor=axis_color2)
        self.magco_slider = Slider(magco_slider_ax, 'Magnetic colatitude', 0, 180, valinit=magco_init, dragging=True)
        phs_slider_ax = self.fig.add_axes([0.25, 0.20, 0.55, 0.03], facecolor=axis_color2)
        self.phs_slider = Slider(phs_slider_ax, "Phase shift", 0, 360, color="purple", valinit=phs_init, dragging=True)
        r_slider_ax = self.fig.add_axes([0.25, 0.25, 0.55, 0.03], facecolor=axis_color2)
        self.r_slider = Slider(r_slider_ax, "Radius", 2, 10, color="green", valinit=r_init, dragging=True)
        az_off_slider_ax = self.fig.add_axes([0.25, 0.30, 0.55, 0.03], facecolor=axis_color2)
        self.az_off_slider = Slider(az_off_slider_ax, "Azimuth Offset", -180, 180, color="orange", valinit=az_off_init,
                                    dragging=True)
        inc_off_slider_ax = self.fig.add_axes([0.25, 0.35, 0.55, 0.03], facecolor=axis_color2)
        self.inc_off_slider = Slider(inc_off_slider_ax, "Inclination Offset", -180, 180, color ="orange",
                                     valinit=inc_off_init, dragging = True)
        gamma1_slider_ax = self.fig.add_axes([0.25, 0.40, 0.55, 0.03], facecolor=axis_color2)
        self.gamma1_slider = Slider(gamma1_slider_ax, "Gamma", 1, 5, color="blue", valinit=gamma1_init, dragging=True)

        self.gamma2_slider_ax = self.fig.add_axes([0.25, 0.45, 0.55, 0.03], facecolor=axis_color2)
        self.gamma2_slider = Slider(self.gamma2_slider_ax, "Gamma2", 1, 5, color="yellow", valinit=gamma2_init,
                                    dragging=True)
        self.gamma2_slider_ax.set_visible(False)
        self.w1_slider_ax = self.fig.add_axes([0.25, 0.95, 0.55, 0.03], facecolor=axis_color2)
        self.w1_slider = Slider(self.w1_slider_ax, "w1", 0, 1, color="yellow", valinit=w1_init, dragging=True)
        self.w1_slider_ax.set_visible(False)
        self.w2_textbox_ax = self.fig.add_axes([0.87, 0.95, 0.1, 0.03])
        self.w2_textbox = TextBox(self.w2_textbox_ax, "w2", initial=str(w2_init))
        self.w2_textbox_ax.set_visible(False)

        """textbox_ax = plt.axes([0.8, 0.10, 0.10, 0.03])  # Position for text box
        self.text_box = TextBox(textbox_ax, '', initial=str(0))
        self.text_box.set_val(round(self.i_slider.val, 2))
        self.text_box.on_submit(self.box_change)"""

        func_radios_ax = self.fig.add_axes([0.03, 0.05, 0.1, 0.20], facecolor = axis_color)
        self.func_radios = RadioButtons(func_radios_ax, ('pencil', 'fan', 'combine'), active=0)
        func_radios_ax.set_title("Select Function", fontsize=12, pad=10)
        
        self.inc_slider.on_changed(self.sphere_slider_on_changed)
        self.pa_slider.on_changed(self.sphere_slider_on_changed)
        self.magco_slider.on_changed(self.sphere_slider_on_changed)
        self.phs_slider.on_changed(self.sphere_slider_on_changed)
        
        self.r_slider.on_changed(self.contour_slider_on_changed)
        self.az_off_slider.on_changed(self.contour_slider_on_changed)
        self.inc_off_slider.on_changed(self.contour_slider_on_changed)
        self.gamma1_slider.on_changed(self.contour_slider_on_changed)

        self.gamma2_slider.on_changed(self.contour_slider_on_changed)
        self.w1_slider.on_changed(self.contour_slider_on_changed)

        self.func_radios.on_clicked(self.func_radios_on_changed)

def main():
    """Testing is done using progscript."""
    pass


if __name__ == '__main__':
    main()