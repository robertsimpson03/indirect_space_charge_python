from pathlib import Path
import numpy as np 
import xobjects as xo
from sim_utils import build_line, add_monitor, add_spacecharge_xfields

n_monitors = 20
n_turns = 50
#n_spacecharge_elements = 20

line = build_line()

#twiss = line.twiss(method='4d', freeze_longitudinal=True)

particles = line.build_particles(x=0, y=0, px=0, py=0)

length = line.get_length()
for i, s in enumerate(np.linspace(0, length, n_monitors, endpoint=False)):
    line = add_monitor(line, particles, s, n_turns, name=i+1)

line = add_spacecharge_xfields(line, 1e9, 10, nemitt_x=1e6, nemitt_y=1e6)
line.build_tracker(_context=xo.ContextCpu(omp_num_threads=10))
line.track(particles, num_turns=n_turns, time=True)

print(line.element_dict['monitor_1'].x)
