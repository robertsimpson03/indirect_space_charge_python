import numpy as np
from gaussian import get_field as g

def get_field(x, y, x0, y0, sx, sy, R):
    """
    Electric field from a 2D Gaussian with circular PEC BC.

    Evaluate the electric field from a 2D Gaussian charge distribution
    using the circular analytical or Bassetti-Erskine semi-analytical
    formula and the method of image charges in a circular, conducting
    boundary.

    Parameters
    ----------
    x : float or array_like
        x-coordinates to evaluate field at.
    y : float or array_like
        y-coordinates to evaluate field at.
    x0 : float
            Beam centre x-coordinate
    y0 : float
            Beam centre y-coordinate
    sx : float
        Gaussian width in the x direction.
    sy : float
        Gaussian width in the y direction.
    R : float
        Radius of the circular boundary

    Returns
    -------
    Ex : ndarray
        x-component of the electric field.
    Ey : ndarray
        y-component of the electric field.
    """

    Ex_ind, Ey_ind = _indirect_field(x, y, x0, y0, sx, sy, R)
    Ex_dir, Ey_dir = g.get_field(x-x0, y-y0, sx, sy)

    Ex = Ex_dir + Ex_ind
    Ey = Ey_dir + Ey_ind

    return Ex, Ey

def _indirect_field(x, y, x0, y0, sx, sy, R):
    R_sqrd = R**2

    Ex = np.zeros_like(x, dtype=float)
    Ey = np.zeros_like(x, dtype=float)
    origin = (x == 0) & (y == 0)
    mask = ~origin
    xm = x[mask]
    ym = y[mask]

    x_sqrd = xm**2
    y_sqrd = ym**2
    inv_r_sqrd = 1 / (x_sqrd + y_sqrd)
    R_on_r_sqrd = R_sqrd*inv_r_sqrd
    txy = 2 * xm * ym
    diff_sqrd = x_sqrd - y_sqrd

    x_image = R_on_r_sqrd * xm
    y_image = R_on_r_sqrd * ym
    Ex_1, Ey_1 = g.get_field(x_image-x0, -(y_image-y0), sx, sy)

    Ex[mask] = (-2*xm + R_on_r_sqrd*(diff_sqrd*Ex_1 - txy*Ey_1))*inv_r_sqrd
    Ey[mask] = (-2*ym + R_on_r_sqrd*(diff_sqrd*Ey_1 + txy*Ex_1))*inv_r_sqrd

    prefactor = 2/R_sqrd
    Ex[origin] = prefactor * x0
    Ey[origin] = prefactor * y0

    return Ex, Ey
