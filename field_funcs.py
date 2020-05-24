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
        
def get_grid(dx,dy,xnum,ynum):
    """
    Retunr xpts,ypts denoting centers of pts on a regular grid
    Args:
        dx,dy: center-center spacing in x,y directions
        xnum, ynum: number of columns, rows
    Return:
        xpts,ypts: lists of x coords,y coords for grid pts
    """
    xpts = []
    ypts = []
    for i in range(xnum):
        for j in range(ynum):
            xpts.append((1+i)*dy - dy*(1+ynum)/2)
            ypts.append((1+j)*dx - dx*(1+xnum)/2)
    return xpts, ypts

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
    
    
def spot_mask(xnum, ynum, a, dx, dy, pts, pos_std=None, phi_std=None, plate=0, aperture=1):
    """
    Warning: this will may not work as expected if xnum or ynum is odd
    Args:
        xnum: # of spots in x (columns)
        ynum " " " " y (rows)
        a: aperture spot radius
        dx: center-center distance in x
        dy: " " " " y
        pts: number of pts in one dimesnsion in the 2D array output. ie output
            mask is pts x pts
        pos_std: std for randomness added to spot centers. should be a decimal 
            representing percentage of 'a', e.g. 0.10 would give normally
            distributed noise with sigma = 0.10*a
        phi_std: std for random phase given to each aperture unit cell. units are in 2*pi and phase is sampled from
            from a normal dist. phi_std = 0.1 would correspond to sigma 0.1*2*pi radians, so there is a 
            +/- 10% spread of phase over the apertures compared to the plate. Note that to create mask where 
            this phase is only applied to the spot, plate must be set to 0. after creating the mask with this function,
            you can then add a constant to offset the transmittance of the whole mask
        plate: 0 by default; plate transmittance
        aperture: 1 by default; aperture transmittance
    Returns: 
        2D array, xarr: binary mask of spots, and 1D array of real space x coordinates.
            The realspace full width of the grid 2*w = (max(xnum,ynum) + 1)*dx
    """

    w = (max(xnum,ynum) + 1)*dx/2 # array real space half-width 
    res = 2*w/pts # real space distance between adjacent pts

    # make subgrid and build a single aperture mask:
    subpts = int(2*a/res) # number of pts to make a side length 2*a
    assert subpts % 2 == 0, "try a slightly different even number of points so that sub-array width is even"
    
    sarr,smidpt,srr,sphi = get_meshgrid(a, subpts, polar=True)
    smask = zeros((subpts,subpts))
    qd3 = smask[:smidpt,:smidpt]
    
    # TODO: fix size of this phase factor
    if phi_std is not None:
        phase = lambda :random.normal(0, 2*pi*phi_std) 
    else:
        phase = lambda :0
    
    for j in range(smidpt):
        for i in range(smidpt):
            qd3[i,j] = int(srr[i,j] < a)
        
    
    smask = from_quadrant3(qd3, smask)
    print(smask.shape)

    # the centroids of the apertures
    xpts, ypts = get_grid(dx,dy,xnum,ynum)
    
    # add noise, optionally
    if pos_std is not None:
        # TODO: add noise from a normal dist of sigma = std*a 
        xpts = array([x + nx for x,nx in zip(xpts,random.normal(0,pos_std,xnum*ynum))])
        ypts = array([y + ny for y,ny in zip(ypts,random.normal(0,pos_std,xnum*ynum))])
    
    # convert centroids to mask indices
    yidcs = [int((y + w)/res) for y in ypts]
    xidcs = [int((x + w)/res) for x in xpts]

    midpt = int(pts/2)
    mask = full((pts, pts), plate, complex)

    # build the mask
    for i in yidcs:
        for j in xidcs:
            mask[i-smidpt:i+smidpt,j-smidpt:j+smidpt] = smask*exp(1j*phase())
            
    # real space coordinates
    xarr = [i*res - w for i in range(pts)]
            
    return mask, xarr