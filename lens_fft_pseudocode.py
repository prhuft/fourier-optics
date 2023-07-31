"""
Lens tranform with FFT - pseudocode
"""

# build the input field, with maximum amplitude of unity
field1 = spot_mask(xnum, ynum, a, dx, dy, pts, pos_std=None, phi_std=None, plate=0, aperture=1)

# make the plane wave broader to increase resolution in Fourier plane
padding = 100 # 100 rows and columns to add
field1 = ones_pad(field1, padding)

# compute the output field with python's fft library
field2 = fftshift(fft2(ifftshift(field1*exp(-1j*k*rr**2*(z2/f - 1)/(2*f)))))

# compute the intensity, normalized to the focal plane intensity
I2_xy = conjugate(field2)*field2
if f - z2 == 0:
    I2xy_max = amax(I2_xy)
I2_xy = I2_xy/I2xy_max 

# Note: I do not plot the intensity beyond the Nyquist spatial frequency. 

