import numpy as np
import xobjects as xo
import xtrack as xt
from scipy.constants import epsilon_0, e
from solvers import direct_field, indirect_field, rectangular_field

k_e = 1.0/(4*np.pi*epsilon_0)

class GaussianSpaceChargeBase(xt.BeamElement):
    iscollective = True  # Allows Python-level track() execution
    _xofields = {
        'element_length': xo.Float64,
        'line_density':   xo.Float64,
        'mean_x':         xo.Float64,
        'mean_y':         xo.Float64,
        'sigma_x':        xo.Float64,
        'sigma_y':        xo.Float64,
    }
    def track(self, particles):
        Ex, Ey = self.get_fields(particles)

        coef = (e*k_e*self.line_density*self.element_length
                /(particles.energy0[0]
                  *(particles.beta0[0]*particles.gamma0[0])**2))

        particles.px += coef * Ex
        particles.py += coef * Ey


class SpaceChargeFree(GaussianSpaceChargeBase):
    def get_fields(self, particles):
        return direct_field(particles.x, particles.y,
                            self.mean_x, self.mean_y,
                            self.sigma_x, self.sigma_y)


class SpaceChargeCircular(GaussianSpaceChargeBase):
    _xofields = {
        'shift_x': xo.Float64,
        'shift_y': xo.Float64,
        'radius':   xo.Float64,
    }

    def get_fields(self, particles):
        Ex_dir, Ey_dir = direct_field(particles.x, particles.y,
                                        self.mean_x, self.mean_y,
                                        self.sigma_x, self.sigma_y)
        Ex_ind, Ey_ind = indirect_field(particles.x, particles.y,
                                        self.mean_x, self.mean_y,
                                        self.sigma_x, self.sigma_y,
                                        self.shift_x, self.shift_y,
                                        self.radius)
        return Ex_dir+Ex_ind, Ey_dir+Ey_ind


class SpaceChargeRectangular(GaussianSpaceChargeBase):
    _xofields = {
        'shift_x': xo.Float64,
        'shift_y': xo.Float64,
        'width':    xo.Float64,
        'height':   xo.Float64,
        'nx':       xo.Field(xo.int64, default=0),
        'ny':       xo.Field(xo.int64, default=0),
    }

    def get_fields(self, particles):
        return rectangular_field(particles.x, particles.y,
                                 self.mean_x, self.mean_y,
                                 self.sigma_x, self.sigma_y,
                                 self.shift_x, self.shift_y,
                                 self.width, self.height,
                                 self.nx, self.ny)
