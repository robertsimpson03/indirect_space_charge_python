#!/usr/bin/env python3

import numpy as np

def get_field(x, y, x0, y0, sx, sy, Lx, Ly, Nx=None, Ny=None):
    """
    Electric field from a 2D Gaussian with rectangular PEC BC.

    Evaluate the electric field from a 2D Gaussian charge distribution
    using a Fourier Series in a rectangular, conducting boundary.

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
    Lx : float
        width in x direction of the rectangular boundary
    Ly : float
        width in y direction of the rectangular boundary
    Nx : int, option
        Number of k-modes in x (default is 30)
    Ny : int, option
        Number of k-modes in y (default is 30)

    Returns
    -------
    Ex : ndarray
        x-component of the electric field.
    Ey : ndarray
        y-component of the electric field.
    """
    
    if Nx==None:
        Nx = np.pi*Lx/sx
    if Ny==None:
        Ny = np.pi*Ly/sy
        
    x0, y0 = x0 + Lx/2, y0 + Ly/2
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    original_shape = x.shape
    x = x.ravel() + Lx/2
    y = y.ravel() + Ly/2

    n = np.arange(1, Nx+1)
    m = np.arange(1, Ny+1)
    kn = np.pi * n / Lx
    km = np.pi * m / Ly
    KN, KM = np.meshgrid(kn, km, indexing='ij')

    norm = 16.0 * np.pi / (Lx * Ly)
    rho_delta_nm = np.sin(KN*x0)*np.sin(KM*y0)
    rho_gaussian_nm = np.exp(-0.5*((KN*sx)**2 + (KM*sy)**2))
    A_nm = norm * rho_delta_nm * rho_gaussian_nm / (KN**2 + KM**2)

    cos_kx = np.cos(np.outer(x, kn))
    sin_kx = np.sin(np.outer(x, kn))
    cos_ky = np.cos(np.outer(y, km))
    sin_ky = np.sin(np.outer(y, km))

    term_y = sin_ky @ A_nm.T
    Ex = - np.sum(cos_kx * term_y * kn, axis=1)

    term_x = sin_kx @ A_nm
    Ey = - np.sum(cos_ky * term_x * km, axis=1)

    return Ex.reshape(original_shape), Ey.reshape(original_shape)
