import numpy as np
import nafflib
import xpart as xp
import xobjects as xo
import xfields as xf

def get_footprint_4d(line, nemitt_x=None, nemitt_y=None, n_turns=256,
                     mode='polar', r_range=(0.1, 6), theta_range=(0.05, np.pi/2 - 0.05),
                     n_r=10, n_theta=10,
                     x_norm_range=(0.1, 6), y_norm_range=(0.1, 6),
                     n_x_norm=10, n_y_norm=10,
                     keep_tracking_data=False):

    if mode == 'polar':
        r = np.linspace(r_range[0], r_range[1], n_r)
        theta = np.linspace(theta_range[0], theta_range[1], n_theta)
        rr, tt = np.meshgrid(r, theta)
        x_norm = rr.flatten() * np.cos(tt.flatten())
        y_norm = rr.flatten() * np.sin(tt.flatten())
    elif mode == 'uniform_action_grid':
        xn = np.linspace(x_norm_range[0], x_norm_range[1], n_x_norm)
        yn = np.linspace(y_norm_range[0], y_norm_range[1], n_y_norm)
        xx, yy = np.meshgrid(xn, yn)
        x_norm = xx.flatten()
        y_norm = yy.flatten()
    else:
        raise ValueError(f"Invalid mode: {mode}")

    particles = xp.build_particles(
        line=line,
        nemitt_x=nemitt_x,
        nemitt_y=nemitt_y,
        x_norm=x_norm,
        y_norm=y_norm,
        method='4d'
    )

    line.build_tracker(_context=xo.ContextCpu())
    line.track(particles, 
               num_turns=n_turns, 
               turn_by_turn_monitor=True)
               #freeze_longitudinal=True)
    mon = line.record_last_track
    
    particles.sort(interleave_lost_particles=True)
    mask = particles.state <= 0
    x = mon.x - np.atleast_2d(mon.x.mean(axis=1)).T
    y = mon.y - np.atleast_2d(mon.y.mean(axis=1)).T
    px = mon.px - np.atleast_2d(mon.px.mean(axis=1)).T
    py = mon.py - np.atleast_2d(mon.py.mean(axis=1)).T
    qx = nafflib.multiparticle_tunes(x, px)
    qy = nafflib.multiparticle_tunes(y, py)
    qx[mask.T] = np.nan
    qy[mask.T] = np.nan
    class Footprint:
        pass

    fp = Footprint()
    fp.qx = qx
    fp.qy = qy
    fp.x_norm = x_norm
    fp.y_norm = y_norm
    if keep_tracking_data:
        fp.tracking_data = {'x': x, 'y': y}

    return fp


def install_spacecharge_frozen(line,
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
