#!/usr/bin/env python3

import numpy as np
from scipy.special import wofz

TOLERANCE = 1e-14
def get_field(x, y, sx, sy):
    """
    Evaluate the electric field from a 2D Gaussian charge distribution.

    Evaluate the electric field from a 2D Gaussian charge distribution
    using the circular analytical or Bassetti-Erskine semi-analytical
    formula.

    Parameters
    ----------
    x : float or array_like
        x-coordinates to evaluate field at.
    y : float or array_like
        y-coordinates to evaluate field at.
    sx : float
        Gaussian width in the x direction.
    sy : float
        Gaussian width in the y direction.

    Returns
    -------
    Ex : ndarray
        x-component of the electric field.
    Ey : ndarray
        y-component of the electric field.
    """

    if abs(sx - sy) < TOLERANCE:
        Ex, Ey = _circular(x, y, sx)
    else:
        if sx > sy:
            Ex, Ey = _elliptical(x, y, sx, sy)
        else:
            Ey, Ex = _elliptical(y, x, sy, sx)

    return Ex, Ey


def _circular(x, y, sigma):
    r_squared = x**2 + y**2
    charge_enclosed = - np.expm1(-r_squared/(2*sigma**2))

    common = 2 * np.divide(charge_enclosed, r_squared,
                           np.zeros_like(r_squared), where=r_squared != 0)

    return x * common, y * common

def _elliptical(x, y, sx, sy):
    Ex, Ey = _bassetti_erskine(x, abs(y), sx, sy)

    Ex = np.asarray(Ex)
    Ey = np.asarray(Ey)
    y = np.asarray(y)

    mask = y < 0
    Ey[mask] *= -1
    return Ex, Ey


def _bassetti_erskine(x, y, sx, sy):
    z = x + 1j*y
    omega = x*sy/sx + 1j*y*sx/sy

    xi_sqrd = (x/sx)**2+(y/sy)**2
    denom = 1 / np.sqrt(2 * (sx**2-sy**2))
    s1 = z * denom
    s2 = omega * denom
    prefactor = 2j*np.sqrt(np.pi) * denom

    field = prefactor * (np.exp(-xi_sqrd/2)*wofz(s2) - wofz(s1))

    return field.real, -field.imag
