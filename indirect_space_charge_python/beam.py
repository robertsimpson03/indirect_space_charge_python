import numpy as np
from scipy.special import wofz

TOLERANCE = 1e-6

class GaussianBeam:
    def __init__(self, x0, y0, sx, sy):
        self.x0 = x0
        self.y0 = y0
        self.sx = sx
        self.sy = sy

    def direct_sc(self, x, y):
        x = x - self.x0
        y = y - self.y0
        return self._direct_sc(x, y)

    def _direct_sc(self, x, y):
        if abs(self.sx - self.sy)/(self.sx + self.sy) < TOLERANCE:
            Ex, Ey = _circular_gaussian(x, y, np.mean([self.sx, self.sy]))
        elif self.sx > self.sy:
            Ex, Ey = _elliptical_gaussian(x, y, self.sx, self.sy)
        else:
            Ey, Ex = _elliptical_gaussian(y, x, self.sy, self.sx)
        return Ex, Ey


class UniformBeam:
    def __init__(self, x0, y0, rx, ry):
            self.x0 = x0
            self.y0 = y0
            self.rx = rx
            self.ry = ry

    def direct_sc(self,x, y):
        x = x - self.x0
        y = y - self.y0
        return self._direct_sc(x, y)

    def _direct_sc(self, x, y):
        rx_sqrd, ry_sqrd = self.rx**2, self.ry**2
        x_sqrd, y_sqrd = x**2, y**2
        inside = (x_sqrd/rx_sqrd + y_sqrd/ry_sqrd) <= 1

        # Elliptical coordinate parameter `s` solution to s^2+bs+c=0
        b = rx_sqrd + ry_sqrd - x_sqrd - y_sqrd
        c = rx_sqrd*ry_sqrd - x_sqrd*ry_sqrd - y_sqrd*rx_sqrd
        s = (-b + np.sqrt(b**2 - 4*c))/2

        denom_in = self.rx + self.ry
        denom_out = s + np.sqrt((rx_sqrd + s)*(ry_sqrd + s))

        Ex = x/np.where(inside, self.rx*denom_in, rx_sqrd + denom_out)
        Ey = y/np.where(inside, self.ry*denom_in, ry_sqrd + denom_out)
        return Ex, Ey


def _circular_gaussian(x, y, sigma):
    r_sqrd = x**2 + y**2
    charge_enclosed = - np.expm1(-r_sqrd/(2*sigma**2))
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
