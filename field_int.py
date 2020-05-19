"""
Module with integrands for the output field of the 4f filtered field array 
Preston Huft
"""

from scipy.integrate import dblquad
from scipy.special import j0, j1
from numpy import *
import matplotlib.pyplot as plt
from time import time

    
class ArrayField2:
    """
    object for computing the field array output from a 4f confiiguration as per Mark's
    simplified trap array proposal. 
    """
    
    def __init__(self, lmbda, f1, f2, a, dx, xnum, dy=None, ynum=None, A0=1, bright=False):
        """
        Constructor for ArrayField2
        
        Args:
            lmbda: wavelength
            f1: first focal length
            f2: second focal length
            a: input aperture radius
            b: Fourier filter radius
            z2: the distance from the second lens
            dx: input grid aperture spacing along x
            dy: optional. if not specified, set dy=dx
            xnum: number of columns of apertures
            ynum: optional. number of rows of apertures. if not specified, xnum=ynum
            bright: optional. bool, specifies whether field is bright
        """
        
        self.lmbda = lmbda
        self.k = 2*pi/lmbda
        self.f1 = f1
        self.f2 = f2
        self.a = a
        self.b = f1*3.8317/(a*self.k) # filter radius, where 3.8317 is the first zero of BesselJ0
        self.dx = dx
        self.xnum = xnum
        self.A0 = 1
        self.bright = bright
        
        if dy is None:
            self.dy = dx
        else:
            self.dy = dy
        
        if ynum is None:
            self.ynum = xnum
        else:
            self.ynum = ynum
            
        self.Lx2 = dx*f2/f1 # output array periodicity in x
        self.Ly2 = dy*f2/f1 # output array periodicity in y
        self.zTalbotx = 2*self.Lx2**2/lmbda # Talbot length x from focal plane
        self.zTalboty = 2*self.Ly2**2/lmbda # Talbot length y from focal plane
        
        self.talbotLengths = {'zTalbotx': self.zTalbotx, 'zTalboty': self.zTalboty}
        
        self.fields = {} # dictionary with field by z value?
            
            
    def field2_xygrid(self, z2, w, pts, name):
        """
        Compute the output field in a plane an axial distance z2 from the second lens
        
        Adds result to 
        
        Args:  
            z2: distance from lens f2
            w: half-width of the grid to be computed
            pts: number of pts to evaluate
            name: the dictionary key to be added to self.fields
        """
        
        xarr2,midpt,xx2,yy2 = ArrayField2.get_meshgrid(w, pts, polar=False)
        field2_z = zeros((pts,pts), complex)
        
        q3 = copy(field2_z[:midpt,:midpt])
        times = []
        avg = 0
        t0 = time()
        for i in range(midpt):
            for j in range(midpt):
                q3[i,j] = self.field2_at_pt(xx2[0,j], yy2[i,0], z2)
            times.append(time()-t0)
            avg = sum(times)/(i+1)
            print(f"y step {i}, time elapsed = {times[i]} s, estimated time remaining: {avg*(midpt - i - 1)} s")
        field2_z = from_quadrant3(q3, field=field2_z)
        print(f"calculated field2 in {(time()-t0)/3600} hrs")
        
        self.field[name] = field2_z
        
        
    def field2_xzgrid(self, numxpts, numzpts, wx, z2i, z2f, y2, name):
        """
        Compute the output field xz plane slice at y
        
        numxpts: number of pts in x
        numztpts: number of pts in z
        wx: halfwidth in x
        z2i, z2f: the inclusive endpts of the z interval
        y2: the y value for the slice
        """
        
        if name in self.fields:
            print("field with name {name} already exists. if this was in err, you may want to interupt the kernel before the field is overwritten")
    
        assert numxpts % 2 == 0, "need even numxpts to be able to reflect field"
        midx = int(numxpts/2)

        xpts2 = linspace(-wx, wx, numxpts)
        zpts2 = linspace(z2i, z2f, numzpts) # plot two Talbot planes back from f2
        
        fieldxz2 = empty((numxpts, numzpts), complex)
        
        t0 = time()
        for j in range(0,numzpts):
            tz0 = time()
            for i in range(midx): # loop 
                fieldxz2[i,j] = self.field2_at_pt(xpts2[i], y2 ,zpts2[j])
            print(f"zstep {j}, time={(time()-tz0)} s")
        print(f"total time={time()-t0}")
        
        fieldxz2[midx:] = flip(fieldxz2[:midx,:], axis=0)
        
        self.fields[name] = fieldxz2
        
    
    def field2_at_pt(self, x2, y2, z2):
        """
        Compute the field at an axial distance z2 from the second lens, at x2,y2 in that plane
        
        Calls the static method field2_int to do this
        
        Args:
            z2: axial distance from lens f2
            x2, y2: planar location of field
        """
        
        field2 =  ArrayField2.field2_int(x2, y2, z2, self.dx, self.xnum, self.ynum, self.a, self.b, 
            dy=self.dy, A0=self.A0, k=self.k, f1=self.f1, f2=self.f2)
        return field2
        
        
    # def int_plot(self, key, args):
        # """
        # Contourf plot of the field
        
        # Args:
            # key: dict key to get the field from self.fields 
            # args: args to be passed optionally to figax
        # """
        
        # try:
            # field = self.fields[key]
        # except KeyError:
            # print(f"{key} is not a valid field key")
            # raise
        
        # Int = conjugate(field)*field
        # Imax = amax(Int)
        # Int /= Imax
        # I2_f2 = 1 - I2_f2
        # f2asym_max
        
        
        
    @staticmethod
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

    @staticmethod
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

        
    @staticmethod
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
        field2 = 4*(A0*a*k)*(dblquad(ArrayField2.re_iint, 1e-6, b, 
                                      lambda x: 0, # keep small enough to obtain good resolution
                                      lambda x: sqrt(b**2 - x**2), 
                                      args=iargs)[0]/(2*pi*f2) 
                              -1j*dblquad(ArrayField2.im_iint, 1e-6, b, 
                                      lambda x: 0, # keep small enough to obtain good resolution 
                                      lambda x: sqrt(b**2 - x**2), 
                                      args=iargs)[0]/(2*pi*f2)) 
        
        return field2
        
        
    @staticmethod
    def from_quadrant3(qd3, field=None):
        """
        construct field with rectangular symmetry given only
        quadrant 3 (i.e., field is mirrored across x axis and
        across y axis)
        
        'qd3' 2D numpy array
        Return 'field' 2D numpy array
        """
        
        xpts, ypts = qd3.shape
        xmpt, ympt = int(xpts/2), int(ypts/2)
        
        if field is None:
            field = zeros((2*xpts, 2*ypts), complex)
        
        field[:midpt, :midpt] = qd3
        # qd4
        # qd1
        field[:midpt, midpt:] = flip(qd3, axis=1) 
        field[midpt:, midpt:] = flip(qd3)
        # qd2
        field[midpt:, :midpt] = flip(qd3, axis=0)
        
        return field
        
        
    @staticmethod
    def get_meshgrid(w, pts, polar=False):
        """
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
            
    @staticmethod
    def get_grid(dx,dy,xnum,ynum):
        """
        returns two lists of points specifying pts on a grid
        dx
        dy
        xnum
        ynum
        """
        xpts = []
        ypts = []
        for i in range(xnum):
            for j in range(ynum):
                xpts.append((1+i)*dy - dy*(1+ynum)/2)
                ypts.append((1+j)*dx - dx*(1+xnum)/2)
        return xpts, ypts
            
            
    @staticmethod
    def figax(roi=None, xlabel=r'x [$\mu$m]', ylabel=r'y [$\mu$m]', aspect='equal'):
        """
        return fig, ax with equal aspect ratio, ideally suited
        """
        fig,ax = plt.subplots()
        if roi is not None:
            ax.set_xlim(-roi,roi)
            ax.set_ylim(-roi,roi)
        if aspect is not None:
            ax.set_aspect(aspect)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return fig, ax