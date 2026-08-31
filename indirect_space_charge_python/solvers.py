import numpy as np
from scipy.special import wofz

TOLERANCE = 1e-10

def direct_field(x, y, x0, y0, sx, sy):
    return _gaussian_field(x-x0, y-y0, sx,sy)


def indirect_field(x, y, x0, y0, sx, sy, mx, my, l):
    x0, y0 = x0 - mx, y0 - my
    x, y = x - mx, y - my
    r_sqrd = x**2 + y**2
    with np.errstate(divide='ignore', invalid='ignore'): # Ignore errors associated with evaluation at the origin
        inv_r_sqrd = 1.0/r_sqrd
        l_sqrd = l**2
        ratio = l_sqrd*inv_r_sqrd

        x_image, y_image = ratio*x, ratio*y
        Ex_dir, Ey_dir = _gaussian_field(x_image-x0, -(y_image-y0), sx, sy)

        diff_sqrd = x**2-y**2
        twoxy = 2*x*y
        Ex_ind = (-2*x + ratio*(diff_sqrd*Ex_dir - twoxy*Ey_dir))*inv_r_sqrd
        Ey_ind = (-2*y + ratio*(diff_sqrd*Ey_dir + twoxy*Ex_dir))*inv_r_sqrd

    origin = (r_sqrd == 0)
    if np.any(origin): # Evaluated at the origin
        Ex_ind[origin] = 2.0*x0/l_sqrd
        Ey_ind[origin] = 2.0*y0/l_sqrd

    return Ex_ind, Ey_ind


def rectangular_field(x, y, x0, y0, sx, sy, mx, my, lx, ly, nx, ny):
    x0, y0 = x0 + lx/2 - mx, y0 + ly/2 - my # Move origin to corner of pipe
    x, y = x + lx/2 - mx, y+ly/2 - my

    nx = nx or int(np.ceil(np.pi*lx/sx))
    ny = ny or int(np.ceil(np.pi*ly/sy))
    kx = np.pi * np.arange(1, nx + 1)/lx
    ky = np.pi * np.arange(1, ny + 1)/ly
    KX, KY = np.meshgrid(kx, ky, indexing='ij')

    norm = 16.0*np.pi/(lx*ly)
    rho_delta = np.sin(KX*x0)*np.sin(KY*y0)
    rho_gaussian = np.exp(-0.5*((KX*sx)**2 + (KY*sy)**2))
    phi = norm*rho_delta*rho_gaussian/(KX**2 + KY**2)

    kx_grid = x[:, None]*kx
    ky_grid = y[:, None]*ky
    term_y = np.sin(ky_grid) @ phi.T
    Ex = -np.sum(np.cos(kx_grid)*term_y*kx, axis=1)
    term_x = np.sin(kx_grid) @ phi
    Ey = -np.sum(np.cos(ky_grid)*term_x*ky, axis=1)

    return Ex, Ey


#######################################################################


def _gaussian_field(x, y, sx, sy):
    if abs(sx - sy)/(sx + sy) < TOLERANCE:
        Ex, Ey = _circular_gaussian(x, y, np.mean([sx, sy]))
    elif sx > sy:
        Ex, Ey = _elliptical_gaussian(x, y, sx, sy)
    else:
        Ey, Ex = _elliptical_gaussian(y, x, sy, sx)
    return Ex, Ey

def _circular_gaussian(x, y, s):
    r_sqrd = x**2 + y**2
    charge_enclosed = - np.expm1(-r_sqrd/(2*s**2))
    common = 2*np.divide(charge_enclosed, r_sqrd,
                         np.zeros_like(r_sqrd), where=r_sqrd != 0)
    return x*common, y*common

def _elliptical_gaussian(x, y, sx, sy):
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
    denom = 1/np.sqrt(2*(sx**2-sy**2))
    s1 = z*denom
    s2 = omega*denom
    prefactor = 2j*np.sqrt(np.pi)*denom

    field = prefactor*(np.exp(-xi_sqrd/2)*wofz(s2) - wofz(s1))
    return field.real, -field.imag
