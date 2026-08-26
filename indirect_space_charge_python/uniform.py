#!/usr/bin/env python3

import numpy as np

def get_field(x, y, a, b):
    """
    Evaluate the E field from a uniform elliptical distribution.

    Evaluate the electric field at a position or for an araay from a 2D
    Uniform elliptical charge distribution using an analytical formula.

    Parameters
    ----------
    x : float or array_like
        x-coordinates to evaluate field at.
    y : float or array_like
        y-coordinates to evaluate field at.
    a : float
        Ellipse x-radius.
    b : float
        Ellipse y-radius.

    Returns
    -------
    Ex : ndarray
        x-component of the electric field.
    Ey : ndarray
        y-component of the electric field.
    """

    Ex = np.zeros_like(x, dtype=np.float64)
    Ey = np.zeros_like(y, dtype=np.float64)

    x_sqrd, y_sqrd = x**2, y**2
    a_sqrd, b_sqrd = a**2, b**2
    inside = (x_sqrd / a_sqrd + y_sqrd / b_sqrd) <= 1
    out = ~inside

    denom_in = a + b
    Ex[inside] = x[inside] / (a * denom_in)
    Ey[inside] = y[inside] / (b * denom_in)

    x_out, y_out = x[out], y[out]
    x_out_sqrd, y_out_sqrd = x_sqrd[out], y_sqrd[out]

    b = a_sqrd + b_sqrd - x_out_sqrd - y_out_sqrd
    c = a_sqrd*b_sqrd - x_out_sqrd*b_sqrd - y_out_sqrd*a_sqrd
    root = (-b + np.sqrt(b**2 - 4*c)) / 2

    denom_out = root + np.sqrt((a_sqrd+root)*(b_sqrd + root))

    Ex[out] = x_out / (a_sqrd + denom_out)
    Ey[out] = y_out / (b_sqrd + denom_out)
    return Ex, Ey
