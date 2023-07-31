"""
Module providing functions for computing Rayleigh-Sommerfeld diffraction

P. Huft

For now, limited to fields with symmetry about xz and yz planes, so that 
only one quadrant of the output field in the x-y plane needs to be computed.
"""

from scipy.integrate import dblquad
from numpy import *
import matplotlib.pyplot as plt
from time import time

# local imports

from field_funcs import get_grid, get_meshgrid, from_quadrant3


class RSDiffract:
    """class for representing a Rayleigh-Sommerfeld diffraction model"""

    def __init__(self, k, hw, field1):
        """
        k: wave number
        hw: real-space half-width. 
        field1: a complex grid representing the field to be propagated; 
                assumed that field input is square
        """
        
        self.k = k
        self.hw = hw
        self.field1 = field1
        self.pts = field1.shape[0]
        self.res = 2*self.hw/self.pts
           
    def re_iint(self, y1, x1, y2, x2, z2):
        """
        the real part of the integrand
        Args:
            y1,x1 the coordinates in the input plane
            y2,x2 the coordinates in the output plane
            z2 the axial distance to propagate
            k: wave number
        """
        
        dx = x2 - x1
        dy = y2 - y1
        r = sqrt(dx**2 + dy**2 + z2**2)
        theta = arctan(sqrt(dx**2 + dy**2)/z2)
        chi = cos(cos(theta)) # obliquity factor: cos(unit(z) dot unit(|r2-r1|))
        
        intgd = (self.field1[int(y1/self.res+0.5), int(x1/self.res+0.5)]
                 *cos(self.k*r)*chi/r)
        
        return intgd


    def im_iint(self, y1, x1, y2, x2, z2):
        """
        the real part of the integrand
        Args:
            y1,x1 the coordinates in the input plane
            y2,x2 the coordinates in the output plane
            z2 the axial distance to propagate
            k: wave number
        """
        
        dx = x2 - x1
        dy = y2 - y1
        r = sqrt(dx**2 + dy**2 + z2**2)
        theta = arctan(sqrt(dx**2 + dy**2)/z2)
        chi = cos(cos(theta)) # obliquity factor: cos(unit(z) dot unit(|r2-r1|))
        
        intgd = (self.field1[int(y1/self.res+0.5), int(x1/self.res+0.5)]
                 *sin(self.k*r)*chi/r)
        
        return intgd

    
    def field2_at_pt(self, x2, y2, z2, rho_sym=True):
        """
        compute the integral for the output field at x2, y2, z2. 
        Integrand is broken up to avoid the pole at x1=y1 = 0; only integrate 
        over quadrant one in the Fourier plane, then multiply the result by 4 
        to account for contributions from the other quadrants.

        Args:
            y1,x1: the coordinates in the input plane
            y2,x2: the coordinates in the output plane
            z2: the axial distance to propagate
            rho_sym = True: limit y integral to sqrt(hw**2 + x**2). else, limit to hw.

        Return:
            'field2': complex output field
        """
        
        x_lim1 = 1e-6 # very small but finite to avoid pole
        y_lim1 = lambda x: 0
        x_lim2 = self.hw
        
        if rho_sym:
            y_lim2 = lambda x: sqrt(self.hw**2 - x**2)
        else:
            y_lim2 = lambda x: self.hw
        
        # unconstrained limits: x in [-hw,hw], y in [-sqrt(hw**2 - x**2), sqrt(hw**2 - x**2)]
        # 1 quadrant: x in [0,hw], y in [0, sqrt(hw**2 - x**2)]. make lower bound finite in either x or y
        
        iargs = (y2, x2, z2) # the arguments after x1,x2 in im_iint, re_iint
        
        # avoid the pole in x
        field2 = 4*self.k*(dblquad(self.re_iint,
                           x_lim1, 
                           x_lim2, 
                           y_lim1,
                           y_lim2, 
                           args=iargs)[0]/(2*pi) 
                           -1j*dblquad(self.im_iint, 
                           x_lim1, 
                           x_lim2, 
                           y_lim1,
                           y_lim2, 
                           args=iargs)[0]/(2*pi))
        
        return field2