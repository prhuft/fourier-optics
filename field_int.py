"""
Module with integrands for the output field of the 4f filtered field array 
Preston Huft
"""

from scipy.integrate import dblquad
from scipy.special import j0, j1
from numpy import sin, cos, sqrt, pi

def re_iint(y1, x1, y2, x2, z2, a, dx, dy, xnum, ynum, A0, k, f1, f2):
    """
    the real part of the integrand
    Args:
        y1,x1 the coordinates in the Fourier plane
        y2,x2 the coordinates in the output plane
        z2 the axial distance from the second lens
        dx: input grid aperture spacing along x
        dy: optional. if not specified, set dy=dx
    """
    
    r1 = sqrt(x1**2 + y1**2)
    
    intgd = (sin(xnum*k*dx*x1/(2*f1))*sin(ynum*k*dy*y1/(2*f1)) \
            /(sin(sin(k*dx*x1/(2*f1))*sin(k*dy*y1/(2*f1))))) \
            *j1(a*r1*k/f1)*cos(k*((z2/f2 - 1)*r1**2 + 2*(x1*x2 + y1*y2))/(2*f2))/r1
    
    return intgd


def im_iint(y1, x1, y2, x2, z2, a, dx, dy, xnum, ynum, A0, k, f1, f2):
    """
    the imaginary part of the integrand
    Args:
        y1,x1 the coordinates in the Fourier plane
        y2,x2 the coordinates in the output plane
        z2 the axial distance from the second lens
        dx: input grid aperture spacing along x
        dy: optional. if not specified, set dy=dx
    """
    
    r1 = sqrt(x1**2 + y1**2)
    
    intgd = - (sin(xnum*k*dx*x1/(2*f1))*sin(ynum*k*dy*y1/(2*f1)) \
              /(sin(sin(k*dx*x1/(2*f1))*sin(k*dy*y1/(2*f1))))) \
              *j1(a*r1*k/f1)*sin(k*((z2/f2 - 1)*r1**2 + 2*(x1*x2 + y1*y2))/(2*f2))/r1
    
    return intgd

    
def field2_int(x2, y2, z2, dx, xnum, ynum, a, b, dy=None, A0=1, k=1, f1=1, f2=1):
    """ 
    compute the integral for the output field of the 4f filtered array with
    integral broken up to avoid the pole at x1=y1 = 0; only integrate over 
    quadrant one in the Fourier plane, then multiply the result by 4 to 
    account for the other quadrants. works by symmetry. 

    Args:
        x2
        y2
        a input aperture radius
        b Fourier filter radius
        z2: the distance from the second lens
        dx: input grid aperture spacing along x
        dy: optional. if not specified, set dy=dx
    Return:
        'field': complex
    """
    
    if dy == None:
        dy = dx
    
    # unconstrained limits: x in [-b,b], y in [-sqrt(b**2 - x**2), sqrt(b**2 - x**2)]
    # 1 quadrant: x in [0,b], y in [0, sqrt(b**2 - x**2)]. make lower bound finite in either x or y
    
    iargs = (y2, x2, z2, a, dx, dy, xnum, ynum, A0, k, f1, f2)
    
    # avoid the pole in x
    field2 = 4*(A0*a*k)*(dblquad(re_iint, 1e-6, b, 
                                  lambda x: 0, # keep small enough to obtain good resolution
                                  lambda x: sqrt(b**2 - x**2), 
                                  args=iargs)[0]/(2*pi*f2) 
                          -1j*dblquad(im_iint, 1e-6, b, 
                                  lambda x: 0, # keep small enough to obtain good resolution 
                                  lambda x: sqrt(b**2 - x**2), 
                                  args=iargs)[0]/(2*pi*f2)) 
    
    return field2