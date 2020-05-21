"""
Quality-of-life functions for my fft and field calculations
"""

## modules. I don't import all from numpy to avoid subtle problems in importing numpy elsewhere
from numpy import linspace, meshgrid, sqrt, arctan2, pi, argmax, flip, zeros
import matplotlib.pyplot as plt

def figax(roi=None, xlabel=None, ylabel=None, aspect='equal'):
    """
    return fig, ax. all params optional
    Args:
        roi: zoom to -roi, roi. None by default
        xlabel: None by default
        ylabel: None by default
        aspect: equal by default
    """
    fig,ax = plt.subplots()
    if roi is not None:
        ax.set_xlim(-roi,roi)
        ax.set_ylim(-roi,roi)
    if aspect is not None:
        ax.set_aspect(aspect)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    return fig, ax
    
def get_meshgrid(w, pts, polar=False):
    """
    Args:
        'w': grid half width
        'pts': pts along one dimension; i.e. the full grid is pts x pts
        'polar': will return rr and phi meshgrid instead of xx,yy
    Returns:
        xarr, midpt, and xx,yy or rr,phi
    """
    midpt = int(pts/2)
    x = linspace(-w, w, pts)
    y = linspace(-w, w, pts)
    xx, yy = meshgrid(x,y, sparse=True)
    if polar:
        rr = sqrt(xx**2 + yy**2)
        phi = arctan2(yy,xx)
        phi[:midpt, :] += 2*pi
        return x,midpt,rr,phi
    else:
        return x,midpt,xx,yy

def justify(arr):
    """
    center an array that goes to zero sufficiently before the endpts
    returns the centered array. 
    
    Args:
        arr: the array to be centered
    Returns:
        a copy of the original array, shifted to be centered
    """
    diffidx = int(argmax(arr) - (len(arr)/2 - 1))
    cp = zeros(len(arr), complex)
    for i in range(0,len(arr)-diffidx):
        cp[i] = arr[i + diffidx]
    return cp

def from_quadrant3(qd3, field=None):
    """
    construct field with rectangular symmetry given only
    quadrant 3 (i.e., field is mirrored across x axis and
    across y axis)
    
    Args;
        'qd3' 2D numpy array
    Returns:
        'field' 2D numpy array
    """
    
    midx, midy = qd3.shape
    
    if field is None:
        field = zeros((2*midx, 2*midy), complex)
    
    field[:midx, :midy] = qd3
    # qd4
    field[:midx, midy:] = flip(qd3, axis=1) 
    # qd1
    field[midx:, midy:] = flip(qd3)
    # qd2
    field[midx:, :midy] = flip(qd3, axis=0)
    
    return field
