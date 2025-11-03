#!/usr/bin/env python3
import numpy as np


class UnitConverter:
    # SI values in l.u.
    m: float
    s: float
    kg: float

    #                                        m   s  kg
    param_dimensions = {'distance':        [ 1,  0,  0],
                        'time':            [ 0,  1,  0],
                        'weight':          [ 0,  0,  1],
                        'viscosity':       [-1, -1,  1],
                        'pressure':        [-1, -2,  1],
                        'energy':          [ 2, -2,  1],
                        'density':         [-3,  0,  1],
                        'concentration':   [-3,  0,  0],
                        'surface_tension': [ 0, -2,  1],
                        'diffusivity':     [ 2, -1,  0]}

    def __init__(self, verbose=True, dx=None, dt=None, kg=None, **kwargs):
        # This can be extended to use other units too

        # Set up the unit matrix and the vector of ratios of the lattice and physical values
        dimensions = [] # Powers of dimensions for each unit: meters, seconds, kilograms
        value_ratios = []
        if (dx is not None):
            dimensions.append([1, 0, 0])
            value_ratios.append(dx)
        if (dt is not None):
            dimensions.append([0, 1, 0])
            value_ratios.append(dt)
        if (kg is not None):
            dimensions.append([0, 0, 1])
            value_ratios.append(1/kg)

        for arg in kwargs:
            values = kwargs[arg]
            if (not hasattr(values, "__len__") or len(values) != 2):
                raise Exception(arg + " requires a 2-tuple containing the physical value and the lattice value.")
            dimensions.append(self.param_dimensions[arg])
            value_ratios.append(values[0]/values[1])

        if (len(dimensions)!=3) or (len(value_ratios)!=3):
            raise Exception("Three values are needed to initialise the unit converter.")

        # Invert the unit matrix to find the values for m, s, kg
        inv_dimensions = np.linalg.inv(dimensions)
        units = np.ones(3)
        for i in range(3):
           for j in range(3):
                units[i] = units[i] * value_ratios[j]**inv_dimensions[i,j] # Matrix multiplication with linear operators of multiplication and raising to a power
        self.m, self.s, self.kg = units

        if (verbose): print(self)


    def __str__(self):
        return "Unit Converter:\n" + \
              f"1 l.u. = {self.m}  metre\n" + \
              f"1 l.u. = {self.s}  second\n" + \
              f"1 l.u. = {self.kg} kilogram"


    def toLattice(self, value, param=None, m=0, s=0, kg=0):
        if (param is not None):
            m  = self.param_dimensions[param][0]
            s  = self.param_dimensions[param][1]
            kg = self.param_dimensions[param][2]
        return value / self.m**m / self.s**s / self.kg**kg


    def toSI(self, value, param=None, m=0, s=0, kg=0):
        if (param is not None):
            m  = self.param_dimensions[param][0]
            s  = self.param_dimensions[param][1]
            kg = self.param_dimensions[param][2]
        return value * self.m**m * self.s**s * self.kg**kg



if (__name__ == "__main__"):
    units = UnitConverter(dx=0.0001, dt=1e-7, surface_tension=[0.0728,0.005])
    #units = UnitConverter(dx=0.0001, dt=4.120879120879121e-07, kg=1/2.472527472527473e-12)
    print("1 second in lattice units:", units.toLattice(1, s=1))
    print("10 meters in lattice units:", units.toLattice(10, m=1))
    print("0.75mm in lattice units:", units.toLattice(0.00075, m=1))
    print("1.81e-5Pa*s of viscosity (air) in lattice units:", units.toLattice(1.81e-5, m=-1, s=-1, kg=1),"tau_air",3*units.toLattice(1.81e-5, m=-1, s=-1, kg=1)+0.5)
    print("1e-2Pa*s of viscosity (water) in lattice units:", units.toLattice(1e-2, m=-1, s=-1, kg=1),"tau_water",3*units.toLattice(1e-2, m=-1, s=-1, kg=1)+0.5)
    print("0.0728mN/ms of surface tension in lattice units:", units.toLattice(0.0728, m=0, s=-2, kg=1))