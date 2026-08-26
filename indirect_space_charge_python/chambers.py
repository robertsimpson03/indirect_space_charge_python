from beam import GaussianBeam, UniformBeam

class NoChamber:
    def space_charge(self, x, y, beam):
        return beam.direct_sc(x,y)

class CircularChamber:
    def __init__(self, radius):
        self.radius = radius

    def space_charge(self, x,y, beam):
        direct_sc_x, direct_sc_y = beam.direct_sc(x,y)
        indirect_sc_x, indirect_sc_y = self._indirect_sc(x,y, beam)
        return direct_sc_x+indirect_sc_x, direct_sc_y+indirect_sc_y

    def _indirect_sc(self, x, y, beam):
        #indirect space charge function
        return 0

class EllipticalChamber:
    def __init__(self, radius_x, radius_y):
        self.radius_x = radius_x
        self.radius_y = radius_y

    def space_charge(self, x,y, beam):
        direct_sc_x, direct_sc_y = beam.direct_sc(x,y)
        indirect_sc_x, indirect_sc_y = self._indirect_sc(x,y, beam)
        return direct_sc_x+indirect_sc_x, direct_sc_y+indirect_sc_y

    def _indirect_sc(self, x, y, beam):
        #indirect space charge function
        return 0

class RectangularChamber:
    def __init__(self, length_x, length_y):
        self.length_x = length_x
        self.length_y = length_y

    def space_charge(self, x,y, beam):
        #assert beam is gaussian
        # beam calculation logic
        return space_charge_x, spacecharge_y
