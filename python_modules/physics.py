import imageprocess as ip
from imageprocess import fft2d, ifft2d
import numpy as np

emass = 510.99906   # electron rest mass in keV
hc = 12.3984244     # Planck's const x speed of light	

def mrad_2_rAng(E0, mrad):
    return mrad * 1e-3 / wavelength(E0)

def rAng_2_mrad(E0, rAng):
    return rAng * wavelength(E0) / 1e-3

def defocusPhase(dz, shape, rA_per_px, E0):
    rho_kernel, _ = ip.polarKernel(shape)
    phase = np.exp( -1j * np.pi * wavelength(E0) * dz *
        (rho_kernel * rA_per_px)**2 )
    return phase

def defocus(arr: np.ndarray, dz, rA_per_px, E0, mode='samespace'):
    phase = defocusPhase(dz, arr.shape, rA_per_px, E0)
    if mode=='samespace':
        return ifft2d( fft2d(arr) * phase )
    elif mode=='reciprocal':
        return ifft2d( arr * phase, b_norm=True )

def wavelength(E0):
    return hc / (E0 * (2*emass + E0))**0.5

def aberration(e0, px_size, shape, ab):
    '''
    Input:
        e0:         energy in keV
        px_size:    pixel size in mrad
        shape:      shape of the output array
        ab:         aberration [c0, c3, a1_a, a1_b] in meter
    Output:
        phase:      complex array of the phase caused by aberration
    '''
    la = wavelength(e0)*1e-10
    rho, phi = ip.polarKernel(shape)
    x, y = ip.cartKernel(shape)
    rho *= px_size*1e-3
    x -= shape[0]//2
    y -= shape[1]//2
    x = x * px_size*1e-3
    y = y * px_size*1e-3
    c0_phase = 0.5 * ab[0] * rho**2
    c3_phase = 0.25 * ab[1] * rho**4
    a1_phase = 0.5 * (ab[2] * (x**2 - y**2) + ab[3] * (x*y))
    phase = -2*np.pi/la * (c0_phase + c3_phase + a1_phase)
    return np.exp(1j*phase)

