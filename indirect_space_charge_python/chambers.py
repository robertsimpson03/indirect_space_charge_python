import numpy as np
from beam import GaussianBeam, UniformBeam

class noChamber:
    def space_charge(self, x, y, beam):
        return beam.direct_sc(x,y)


class CircularChamber:
    def __init__(self, radius):
        self.radius = radius

    def space_charge(self, x,y, beam):
        Ex_dir, Ey_dir = beam.direct_sc(x,y)
        Ex_ind, Ey_ind = self._indirect_sc(x, y, beam)
        return Ex_dir+Ex_ind, Ey_dir+Ey_ind

    def _indirect_sc(self, x, y, beam):
        r_sqrd = x**2 + y**2
        # Ignore errors associated with evaluation at the origin
        with np.errstate(divide='ignore', invalid='ignore'):
            inv_r_sqrd = 1.0/r_sqrd
            radius_sqrd = self.radius**2
            ratio = radius_sqrd*inv_r_sqrd

            x_image, y_image = ratio*x, ratio*y
            Ex_dir, Ey_dir = beam._direct_sc(x_image-beam.x0, -(y_image-beam.y0))

            diff_sqrd = x**2-y**2
            twoxy = 2*x*y
            Ex_ind = (-2*x + ratio*(diff_sqrd*Ex_dir - twoxy*Ey_dir))*inv_r_sqrd
            Ey_ind = (-2*y + ratio*(diff_sqrd*Ey_dir + twoxy*Ex_dir))*inv_r_sqrd

        # Calculate formula for at the origin
        origin = (r_sqrd == 0)
        if np.any(origin):
            Ex_ind[origin] = 2.0*beam.x0/radius_sqrd
            Ey_ind[origin] = 2.0*beam.y0/radius_sqrd

        return Ex_ind, Ey_ind


class RectangularChamber:
    def __init__(self, length_x, length_y):
        self.lx = length_x
        self.ly = length_y

    def space_charge(self, x,y, beam, nx=None, ny=None):
        assert isinstance(beam, GaussianBeam), \
            f"RectangularChamber requires GaussianBeam"

        lx, ly = beam.lx, beam.ly
        # Move origin to corner of pipe
        x0, y0 = beam.x0 + lx/2, beam.y0 + ly/2
        x, y = x + lx/2, y+ly/2

        nx = nx or int(np.ceil(np.pi*lx/beam.sx))
        ny = ny or int(np.ceil(np.pi*ly/beam.sy))
        kx = np.pi * np.arange(1, nx + 1)/lx
        ky = np.pi * np.arange(1, ny + 1)/ly
        KX, KY = np.meshgrid(kx, ky, indexing='ij')

        norm = 16.0*np.pi/(lx*ly)
        rho_delta = np.sin(KX*x0)*np.sin(KY*y0)
        rho_gaussian = np.exp(-0.5*((KX*beam.sx)**2 + (KY*beam.sy)**2))
        phi = norm*rho_delta*rho_gaussian/(KX**2 + KY**2)

        kx_grid = x[:, None]*kx
        ky_grid = y[:, None]*ky
        term_y = np.sin(ky_grid) @ phi.T
        Ex = -np.sum(np.cos(kx_grid)*term_y*kx, axis=1)
        term_x = np.sin(kx_grid) @ phi
        Ey = -np.sum(np.cos(ky_grid)*term_x*ky, axis=1)

        return Ex, Ey


"""
not implemented yet
A solution may invovle a conformal mapping from the circular chamber
class EllipticalChamber:
    def __init__(self, radius_x, radius_y):
        self.radius_x = radius_x
        self.radius_y = radius_y

    def space_charge(self, x,y, beam):
        Ex_dir, Ey_dir = beam.direct_sc(x,y)
        Ex_ind, Ey_ind = self._indirect_sc(x,y, beam)
        return Ex_dir+Ex_ind, Ey_dir+Ey_ind

    def _indirect_sc(self, x, y, beam):
        #indirect space charge function
        return 0
"""
