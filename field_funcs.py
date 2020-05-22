"""
Quality-of-life functions for my fft and field calculations
"""

## modules. I don't import all from numpy to avoid subtle problems in importing numpy elsewhere
from numpy import linspace, meshgrid, sqrt, arctan2, pi, argmax, flip, zeros, append
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


def zero_pad(field, thiccness):
    """
    pad the field on each edge with zeros
    
    Args:
        field: 2D array to be padded
        thiccness: number of rows and columns of zeros to be added to field
    Returns:
        2D array of shape (field.shape[0]+2*thiccness, field.shape[1]+2*thiccness)
    """
    
    rows, cols = field.shape
    # pad left/right edges
    field = append(field, zeros((rows, thiccness)), axis=1)
    field = append(zeros((rows, thiccness)), field, axis=1)
    # pad top/bottom edges
    cols = field.shape[1]
    field = append(field, zeros((thiccness, cols)), axis=0)
    field = append(zeros((thiccness, cols)), field, axis=0)
    
    return field
    
def unpad(field, thiccness):
    """
    unpad the field on each edge given the padding thiccness
    
    Args:
        field: 2D array to be padded
        thiccness: number of rows and columns to be removed
    Returns:
        2D array of shape (field.shape[0]-2*thiccness, field.shape[1]-2*thiccness)
    """
    
    field = field[thiccness:,:]
    field = field[:-thiccness,:]
    field = field[:,thiccness:]
    field = field[:,:-thiccness]

    return field
    
def circ_mask(rgrid, radius):
    """
    create a mask that is 1 within some radius and 0 otherwise
    Args:
        rgrid: the radius grid to use for evaluating the radius equality
        radius: the radius, in the same units as rgrid
    Return:
        mask: the binary mask, a 2D array with same dimensions as rgrid
    """   
    
    rows,cols = rgrid.shape
    
    mask = zeros((rows,cols))
    for i in range(cols):
        for j in range(rows):
            if rgrid[i,j] < radius:
                mask[i,j] = 1
                
    return mask