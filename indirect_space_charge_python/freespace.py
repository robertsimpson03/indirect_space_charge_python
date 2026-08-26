from gaussian import get_field as g

def get_field(x, y, x0, y0, sx, sy):
    """
    Electric field from a 2D Gaussian in free space.

    Evaluate the electric field from a 2D Gaussian charge distribution
    using the circular analytical or Bassetti-Erskine semi-analytical
    formula in free space.

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
    sigma_x : float
        Gaussian width in the x direction.
    sigma_y : float
        Gaussian width in the y direction.

    Returns
    -------
    Ex : ndarray
        x-component of the electric field.
    Ey : ndarray
        y-component of the electric field.

    """

    Ex, Ey = g.get_field(x-x0, y-y0, sx, sy)
    return Ex, Ey
