"""
Module providing functions for computing Rayleigh-Sommerfeld diffraction

P. Huft

For now, limited to radially symmetric fields.
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
            
    # TODO - amend these methods
    # def field2_xygrid(self, z2, w, pts, name):
        # """
        # Compute the field after an axial distance z2. 
        
        # (z1 is the supposed starting plane)
            
        # Args:  
            # z2: axial distance to propagate field
            # w: half-width of the grid to be computed
            # pts: number of pts to evaluate
            # name: the dictionary key to be added to self.fields
        # """
        
        # xarr2,midpt,xx2,yy2 = ArrayField2.get_meshgrid(w, pts, polar=False)
        # field2_z = zeros((pts,pts), complex)
        
        # q3 = copy(field2_z[:midpt,:midpt])
        # times = []
        # avg = 0
        # t0 = time()
        # for i in range(midpt):
            # for j in range(midpt):
                # q3[i,j] = self.field2_at_pt(xx2[0,j], yy2[i,0], z2)
            # times.append(time()-t0)
            # avg = sum(times)/(i+1)
            # print(f"y step {i}, time elapsed = {times[i]} s, estimated time remaining: {avg*(midpt - i - 1)} s")
        # field2_z = ArrayField2.from_quadrant3(q3, field=field2_z)
        # print(f"calculated field2 in {(time()-t0)/3600} hrs")
        
        # self.fields[name] = field2_z
        
            
    # def field2_xzgrid(self, numxpts, numzpts, wx, z2i, z2f, y2, name):
        # """
        # Compute the output field xz plane slice at y
        
        # numxpts: number of pts in x
        # numztpts: number of pts in z
        # wx: halfwidth in x
        # z2i, z2f: the inclusive endpts of the z interval
        # y2: the y value for the slice
        # """
        
        # if name in self.fields:
            # print("field with name {name} already exists. if this was in err, you may want to interupt the kernel before the field is overwritten")

        # assert numxpts % 2 == 0, "need even numxpts to be able to reflect field"
        # midx = int(numxpts/2)

        # xpts2 = linspace(-wx, wx, numxpts)
        # zpts2 = linspace(z2i, z2f, numzpts) # plot two Talbot planes back from f2
        
        # fieldxz2 = empty((numxpts, numzpts), complex)
        
        # t0 = time()
        # for j in range(0,numzpts):
            # tz0 = time()
            # for i in range(midx): # loop 
                # fieldxz2[i,j] = self.field2_at_pt(xpts2[i], y2 ,zpts2[j])
            # print(f"zstep {j}, time={(time()-tz0)} s")
        # print(f"total time={time()-t0}")
        
        # fieldxz2[midx:] = flip(fieldxz2[:midx,:], axis=0)
        
        # self.fields[name] = fieldxz2 
           
    @staticmethod
    def re_iint(y1, x1, y2, x2, z2):
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
        
        intgd = self.field1[y1/self.res, x1/self.res]*cos(self.k*r)*chi/r
        
        return intgd

    @staticmethod
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
        
        intgd = self.field1[y1/self.res, x1/self.res]*sin(self.k*r)*chi/r
        
        return intgd

    
    def field2_at_pt(self, x2, y2, z2):
        """
        compute the integral for the output field at x2, y2, z2. 
        Integrand is broken up to avoid the pole at x1=y1 = 0; only integrate 
        over quadrant one in the Fourier plane, then multiply the result by 4 
        to account for contributions from the other quadrants.

        Args:
            y1,x1 the coordinates in the input plane
            y2,x2 the coordinates in the output plane
            z2 the axial distance to propagate

        Return:
            'field2': complex output field
        """
        
        if dy == None:
            dy = dx
        
        # unconstrained limits: x in [-b,b], y in [-sqrt(b**2 - x**2), sqrt(b**2 - x**2)]
        # 1 quadrant: x in [0,b], y in [0, sqrt(b**2 - x**2)]. make lower bound finite in either x or y
        
        iargs = (y2, x2, z2 # the arguments after x1,x2 in im_iint, re_iint
        
        # avoid the pole in x
        field2 = 4*self.k*(dblquad(RSDiffract.re_iint, 1e-6, self.hw, 
                                      lambda x: 0, # keep small enough to obtain good resolution
                                      lambda x: sqrt(self.hw**2 - x**2), 
                                      args=iargs)[0]/(2*pi) 
                              -1j*dblquad(RSDiffract.im_iint, 1e-6, self.hw, 
                                      lambda x: 0, # keep small enough to obtain good resolution 
                                      lambda x: sqrt(self.hw**2 - x**2), 
                                      args=iargs)[0]/(2*pi))
        
        return field2