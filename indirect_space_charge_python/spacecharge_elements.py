import numpy as np
import xobjects as xo
import xtrack as xt
from scipy.constants import epsilon_0
from solvers import direct_field, indirect_field, rectangular_field

k_e = 1.0/(4*np.pi*epsilon_0)

class GaussianSpaceChargeBase(xt.BeamElement):
    iscollective = True  # Allows Python-level track() execution
    _xofields = {
        'element_length': xo.Float64,
        'mean_x':         xo.Float64,
        'mean_y':         xo.Float64,
        'sigma_x':        xo.Float64,
        'sigma_y':        xo.Float64,
        'line_density':   xo.Float64,
    }
    def track(self, particles):
        Ex, Ey = self.get_fields(particles)

        coef = (k_e * self.line_density * self.element_length
                /(particles.energy0[0] * (particles.beta0[0]
                                         *particles.gamma0[0])**2))

        particles.px += coef * Ex
        particles.py += coef * Ey


class SpaceChargeFree(GaussianSpaceChargeBase):
    def get_fields(self, particles):
        return direct_field(particles.x, particles.y,
                            self.mean_x, self.mean_y,
                            self.sigma_x, self.sigma_y)


class SpaceChargeCircular(GaussianSpaceChargeBase):
    _xofields = {
        'centre_x':   xo.Float64,
        'centre_y':   xo.Float64,
        'radius': xo.Float64,
    }

    def __init__(self, centre_x=0, centre_y=0, radius=None,
                    a=None, b=None, **kwargs):
        if radius is None:
            if a is not None and a==b:
                radius=a
            else:
                raise ValueError("Ellipse not implemented: circle only")

        super().__init__(
            centre_x = centre_x,
            centre_y = centre_y,
            radius = radius,
            **kwargs
        )

    def get_fields(self, particles):
        Ex_dir, Ey_dir = direct_field(particles.x, particles.y,
                                        self.mean_x, self.mean_y,
                                        self.sigma_x, self.sigma_y)
        Ex_ind, Ey_ind = indirect_field(particles.x, particles.y,
                                        self.mean_x, self.mean_y,
                                        self.sigma_x, self.sigma_y,
                                        self.centre_x, self.centre_y,
                                        self.radius)
        return Ex_dir+Ex_ind, Ey_dir+Ey_ind


class SpaceChargeRectangular(GaussianSpaceChargeBase):
    _xofields = {
        'centre_x':   xo.Float64,
        'centre_y':   xo.Float64,
        'width': xo.Float64,
        'height': xo.Float64,
        'nx':       xo.Int64,
        'ny':       xo.Int64
    }

    def __init__(self, centre_x=None, centre_y=None, width=None,
                 height=None, min_x=None, max_x=None, min_y=None,
                 max_y=None, nx=None, ny=None, **kwargs):
        if centre_x is not None and width is not None:
            pass
        elif min_x is not None and max_x is not None:
            centre_x = (max_x + min_x)/2
            width = max_x - min_x
        else:
            raise ValueError("Pipe Geometry must be provided")

        if centre_y is not None and height is not None:
            pass
        elif min_y is not None and max_y is not None:
            centre_y = (max_y + min_y)/2
            height = max_y - min_y
        else:
            raise ValueError("Pipe Geometry must be provided")

        nx = nx or 0
        ny = ny or 0

        super().__init__(
            centre_x = centre_x,
            centre_y = centre_y,
            width = width,
            height = height,
            nx = nx,
            ny = ny,
            **kwargs
        )

    def get_fields(self, particles):
        return rectangular_field(particles.x, particles.y,
                                 self.mean_x, self.mean_y,
                                 self.sigma_x, self.sigma_y,
                                 self.centre_x, self.centre_y,
                                 self.width, self.height,
                                 self.nx, self.ny)
