import numpy as np
import xtrack as xt

from .spacecharge_elements import SpaceChargeFree, SpaceChargeCircular, SpaceChargeRectangular

PREFIX = 'spacecharge_ele_' # How are space charge elements names
TOL = 1e-2 # For determining is an Elliptical chaber is circular 

def install_spacecharge_frozen(line,
                               number_of_particles,
                               num_spacecharge_interactions,
                               nemitt_x, nemitt_y,
                               indirect_sc = True,
                               particle_ref=None,
                               node_s_list=None,
                               _enlargement=1):
    """Install frozen, coasting, indirect spacecharge elements into an 
       Xtrack line.

      Parameters
      ----------
      line : xtrack.Line
          The beam line where spacecharge elements will be installed.
      number_of_particles : float
          Total number of particles in the beam (used to calculate coasting beam
          line density).
      num_spacecharge_interactions : int
          Number of spacecharge interaction nodes to distribute along the line.
      nemitt_x : float
          Normalized horizontal emittance [m rad].
      nemitt_y : float
          Normalized vertical emittance [m rad].
      indirect_sc : bool, optional
          Flag to include indirect space charge effects. Defaults to True.
      particle_ref : xtrack.Particles, optional
          Reference particle. If None, it is inferred from `line.particle_ref`.
          Defaults to None.
      node_s_list : array-like, optional
          Custom longitudinal s-positions for the spacecharge interaction nodes.
          If None, nodes are evenly spaced. Defaults to None.
      _enlargement : float, optional
          Scale factor applied to the aperture dimensions. Defaults to 1.

      Returns
      -------
      xtrack.Line
          The modified line with the frozen spacecharge elements built and inserted.
    """
    line.discard_tracker()

    if particle_ref is None:
        particle_ref = line.particle_ref
        assert particle_ref is not None
    line_length = line.get_length()
    line_density = number_of_particles/line_length # Coasting beam
    emitt_x = nemitt_x/(particle_ref.beta0*particle_ref.gamma0)
    emitt_y = nemitt_y/(particle_ref.beta0*particle_ref.gamma0)

    if node_s_list is None:
        node_s_list = np.linspace(0, line_length, num_spacecharge_interactions+1)[:-1]
    line, node_name_list = _create_nodes(line, node_s_list) 
    node_length_list = _get_node_length_list(node_s_list, line_length)

    line.build_tracker()
    tw_at_sc = line.twiss4d(particle_ref=particle_ref).rows[node_name_list]

    for i, node_s in enumerate(node_s_list): # Break down in to smaller functions
        node_length = node_length_list[i]

        trans_params = _get_trans_params(tw_at_sc, i, emitt_x, emitt_y)
        long_params = {'element_length': node_length, 'line_density': line_density}

        if indirect_sc == True:
            aper_type, aper_params = _get_aper_params(line, node_s, node_length, line_length, _enlargement)
            if aper_type == 'circular':
                sc_element = SpaceChargeCircular(**(trans_params|long_params|aper_params))
            elif aper_type == 'rectangular':
                sc_element = SpaceChargeRectangular(**(trans_params|long_params|aper_params))
            else:
                sc_element = SpaceChargeFree(**(trans_params|long_params))
        else:
            sc_element = SpaceChargeFree(**(trans_params|long_params))


        line.element_dict[node_name_list[i]] = sc_element

    return line


def _create_nodes(line, node_s_list,):
    node_name_list = [f'{PREFIX}{i}' for i in range(len(node_s_list))]
    insertions = [
            (s, [(name, xt.Marker())]) 
            for s, name in zip(node_s_list, node_name_list)
    ]
    line._insert_thin_elements_at_s(insertions)
    return line, node_name_list

def _get_node_length_list(node_s_list, line_length):
    node_length_list = np.empty_like(node_s_list)
    node_length_list[:-1] = np.diff(node_s_list)
    node_length_list[-1] = line_length - np.sum(node_length_list[:-1])
    return node_length_list

def _get_trans_params(tw, i, emitt_x, emitt_y):
    mean_x = tw['x'][i]
    mean_y = tw['y'][i]
    sigma_x = np.sqrt(tw['betx'][i]*emitt_x)
    sigma_y = np.sqrt(tw['bety'][i]*emitt_y)
    return {'mean_x': mean_x, 'mean_y':mean_y,
            'sigma_x': sigma_x, 'sigma_y': sigma_y}

def _get_aper_params(line, s, node_length, line_length, _enlargement):
    tt = line.get_table(attr=True)
    if s+node_length < line_length:
        mask = (tt.s >= s) & (tt.s <= s + node_length)
    else: # If the section being search wraps around to the start of the line
        mask = (tt.s >= s) | (tt.s <= s + node_length - line_length)
    local_elements = tt.rows[mask]
    
    for name in local_elements.name:
        element = line[name]

        if isinstance(element, xt.LimitEllipse):
            if abs(element.a-element.b)/(element.a+element.b) < TOL:
                return "circular", {'shift_x': element.shift_x,
                                    'shift_y': element.shift_y,
                                    'radius': _enlargement*element.a}
            else:
                return "elliptical", {}

        if isinstance(element, xt.LimitRect):
            shift_x = (element.max_x + element.min_x) / 2 + element.shift_x
            shift_y = (element.max_y + element.min_y) / 2 + element.shift_y
            width = _enlargement*(element.max_x - element.min_x)
            height = _enlargement*(element.max_y - element.min_y)

            return "rectangular", {'shift_x': shift_x, 'shift_y': shift_y,
                                   'width': width, 'height': height}
    return None, {} # If no suitable aperture is found
