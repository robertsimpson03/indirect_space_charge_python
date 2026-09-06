import numpy as np

import xobjects as xo
from xtrack.progress_indicator import progress
import xtrack as xt
import xpart as xp

from spacecharge_elements import SpaceChargeFree, SpaceChargeCircular, SpaceChargeRectangular

def install_spacecharge_frozen(line,
                               number_of_particles,
                               num_spacecharge_interactions,
                               nemitt_x, nemitt_y,
                               particle_ref=None,
                               s_spacecharge=None,
                               prefix='spacecharge_'):

    line.discard_tracker() # as we will be changing element types

    if particle_ref is None:
        particle_ref = line.particle_ref
        assert particle_ref is not None

    if s_spacecharge is None:
        s_spacecharge_target = np.linspace(0, line.get_length(),
                                    num_spacecharge_interactions+1)[:-1]
    else:
        s_spacecharge_target = s_spacecharge

    # Create spacecharge elements (dummy markers)
    sc_names = [f'{prefix}{ii}' for ii in range(len(s_spacecharge_target))]
    insertions = [(s, [(name, xt.Marker())]) for s, name in zip(s_spacecharge_target, sc_names)]
    line._insert_thin_elements_at_s(insertions)

    tt = line.get_table()
    s_spacecharge = tt.rows[sc_names].s
    sc_lengths = np.empty_like(s_spacecharge)
    sc_lengths[:-1] = np.diff(s_spacecharge)
    sc_lengths[-1] = line.get_length() - np.sum(sc_lengths[:-1])

    line.build_tracker()
    tw_at_sc = line.twiss4d(particle_ref=particle_ref).rows[sc_names]

    line_density = number_of_particles/line.get_length()
    emitt_x = nemitt_x/(particle_ref.beta0*particle_ref.gamma0)
    emitt_y = nemitt_y/(particle_ref.beta0*particle_ref.gamma0)
    for ii, name in enumerate(sc_names):
        ss = s_spacecharge[ii]
        length = sc_lengths[ii]

        mean_x = tw_at_sc['x'][ii]
        mean_y = tw_at_sc['y'][ii]
        sigma_x = np.sqrt(tw_at_sc['betx'][ii]*emitt_x)
        sigma_y = np.sqrt(tw_at_sc['bety'][ii]*emitt_y)
        length = sc_lengths[ii]

        aper_type, aper_params = _get_aper_at_s(line, ss, length)
        base_params = {
            'element_length': length,
            'line_density': line_density,
            'mean_x': mean_x, 'mean_y': mean_y,
            'sigma_x': sigma_x, 'sigma_y': sigma_y
        }

        if aper_type == 'circular':
            sc_element = SpaceChargeCircular(**base_params, **aper_params)
        elif aper_type == 'rectangular':
            sc_element = SpaceChargeRectangular(**base_params, **aper_params)
        else:
            sc_element = SpaceChargeFree(**base_params)

        line.element_dict[name] = sc_element


def _get_aper_at_s(line, s, length):
    tt = line.get_table(attr=True)

    mask = (tt.s >= s) & (tt.s <= s + length)
    local_apertures = tt.rows[mask]

    for name in local_apertures.name:
        aperture = line[name]
        if isinstance(aperture, xt.LimitEllipse):
            if aperture.a_x==aperture.a_y:
                return "circular", {
                    'shift_x': aperture.shift_x,
                    'shift_y': aperture.shift_y,
                    'radius': aperture.a_x,
                }
            else:
                return "elliptical", {}

        if isinstance(aperture, xt.LimitRect):
            shift_x = (aperture.max_x + aperture.min_x) / 2 + aperture.shift_x
            shift_y = (aperture.max_y + aperture.min_y) / 2 + aperture.shift_y
            width = aperture.max_x - aperture.min_x
            height = aperture.max_y - aperture.min_y

            return "rectangular", {
                'shift_x': shift_x,
                'shift_y': shift_y,
                'width': width,
                'height': height,
            }
    return None, None
