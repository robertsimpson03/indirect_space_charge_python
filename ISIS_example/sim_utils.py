from pathlib import Path
import time
import h5py

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.constants import e, m_p, epsilon_0
k_e = 1/(4*np.pi*epsilon_0)

import xtrack as xt
import xpart as xp
import xobjects as xo
import xfields as xf

from cpymad.madx import Madx


def build_line(directory=None,
               slices=4, thick=False, install_apertures=True):
    if directory==None:
        module_directory = Path(__file__).resolve().parent 
        directory = module_directory / '../Lattice_Files/02_Aperture_Lattice/'
        directory = Path(directory).resolve()

    print(str(directory / 'ISIS.injected_beam'))
    madx = Madx(stdout=False)
    madx.call(str(directory / 'ISIS.injected_beam'))
    madx.call(str(directory / 'ISIS.elements'))
    madx.call(str(directory / 'ISIS.strength'))
    madx.call(str(directory / 'ISIS.sequence'))
    madx.call(str(directory / 'ISIS.aperture'))
    madx.use('synchrotron')
    madx.command.select(flag='makethin', slice=slices, thick=thick)
    madx.command.makethin(sequence='synchrotron', style='teapot',
                          makedipedge=True)

    line = xt.Line.from_madx_sequence(
                madx.sequence.synchrotron,
                install_apertures=install_apertures
                )
    line.set_particle_ref('proton', p0c=0.37033168 * 1e9)
    return line


def add_dipole(line, k_list, s_list, mode='normal'):
    s_list = np.atleast_1d(s_list)
    if not hasattr(k_list, '__len__'):
        k_list = k_list * np.ones_like(s_list)
    kicks = []
    for s_list, k_list in zip(s_list, k_list):
        name = f'dipole_at_{s_list:.3f}'
        if mode=='normal':
            dipole = xt.Multipole(knl=[k_list])
        else:
            dipole = xt.Multipole(ksl=[k_list])
        kicks.append(line.env.place(name, obj=dipole, at=s_list))
    line.insert(kicks)
    return line

def add_monitor(line, particles, s, n_turns, name='untitled'):
    n_particles = len(particles.particle_id)
    mon = xt.ParticlesMonitor(start_at_turn=0, stop_at_turn=n_turns, num_particles=n_particles)
    monitor = line.env.place(f'monitor_{name}', obj=mon, at=s)
    line.insert(monitor)
    return line

def add_spacecharge_xfields(line,
                            number_of_particles,
                            num_spacecharge_interactions,
                            nemitt_x, nemitt_y,
                            particle_ref=None,
                            s_spacecharge=None,
                            prefix='spacecharge_'):

    line_density = number_of_particles/line.get_length()
    sigma_z_fake = 1e16 # Arbitrary must be >>length
    number_of_particles_fake = line_density*np.sqrt(2*np.pi)*sigma_z_fake

    lprofile = xf.LongitudinalProfileQGaussian(
            number_of_particles= number_of_particles_fake,
            sigma_z= sigma_z_fake
        )
    xf.install_spacecharge_frozen(line,
        longitudinal_profile=lprofile,
        nemitt_x=nemitt_x,
        nemitt_y=nemitt_y,
        sigma_z=sigma_z_fake,
        num_spacecharge_interactions=num_spacecharge_interactions,
        delta_rms=0
        )

    return line
