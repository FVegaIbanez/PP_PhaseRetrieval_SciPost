import numpy as np
import tools.imageprocess as ip

emass = 510.99906   # electron rest mass in keV
hc = 12.3984244     # Planck's const x speed of light	

def mrad_2_rAng(E0, mrad):
    return mrad * 1e-3 / wavelength(E0)

def rAng_2_mrad(E0, rAng):
    return rAng * wavelength(E0) / 1e-3

def wavelength(E0):
    return hc / (E0 * (2*emass + E0))**0.5

def aberration(e0, px_size, shape, coef, basis='power', conv_angle=None):
    '''
    Generating a complex array describing aberration in power law or Zernike polynomial.

    Order of the aberration coefficient for power series:
    Krivanek | Haider | Description
    -------------------------------
    C0,1     |        | Beam shift
    C1,2     | A1     | Twofold axial astigamtism
    C1,0     | C1     | Defocus (overfocus positive)
    C2,3     | A2     | Threefold axial astigmatism
    C2,1     | B2     | Second-order axial coma
    C3,4     | A3     | Fourfold axial astigmatism
    C3,2     | S3     | Twofold astigmatism of Cs (star aberration)
    C3,0     | C3     | Third-order spherical aberration
    C4,5     | A4     | 
    C4,1     | B4     |
    C4,3     | D4     |
    C5,4     | R5     |
    C5,2     | S5     |
    C5,0     | C5     |
    C5,6     | A5     |
    C6,1     | B6     |
    C6,3     | D6     |
    C6,5     | F6     |
    C6,7     | A6     |
    C7,0     | C7     |
    C7,2     | S7     |
    C7,4     | R7     |
    C7,6     | G7     |
    C7,8     | A7     |

    Aberration coefficient for normalized Zernike polynomial is described by (j, n, m)

    Parameter
    ---------
        e0:         Energy in keV
        px_size:    Pixel size in mrad
        shape:      Shape of the output array
        coef:       Aberration coefficients in the sequence of increasing order.
        basis:      Which basis to decompose aberration, power series or 
                    Zernike polynomial. Default: power
        conv_angle: (only for basis `zernike`) Convergence angle
    
    Return
    ------
        phase:      A complex array of the aberration
    '''
    def normzernike_r(n,m,rho_arr):
        if (n-m)//2 != 0:
            return 0
        s = 0
        s_lim = (n-m)//2
        r_arr = np.zeros(rho_arr.shape)
        while s <= s_lim:
            r_arr += \
                (-1)**s * np.math.factorial(n-s) * rho_arr**(n-2*s) /\
                np.math.factorial(s) * np.math.factorial((n+m)//2-s) *\
                np.math.factorial((n-m)//2-s)
        r_arr[rho_arr==0] = 1
        r_arr[r_arr>1] = 0
        return r_arr

    la = wavelength(e0)*1e-10
    aberration = np.zeros(shape)

    if basis == 'power':
        x, y = ip.cartKernel(shape)
        x = (x-shape[0]//2) * px_size *1e-3
        y = (y-shape[1]//2) * px_size *1e-3
        omega = (x + 1j * y)
        omegabar = (x - 1j * y)
        const_coef = [
            1, 1/2, 1/2, 1/3, 1, 1/4,
            1, 1/4, 1/5, 1, 1, 1, 1, 
            1/6, 1/6, 1, 1, 1, 1/7, 
            1/8, 1, 1, 1, 1/8]
        omega_power = [
            (0,1), (0,2), (1,1), (0,3), (2,1),
            (0,4), (3,1), (2,2), (0,5), (3,2),
            (4,1), (5,1), (4,2), (3,3), (0,6),
            (4,3), (5,2), (6,1), (0,7), (4,4),
            (5,3), (6,2), (7,1), (0,8)]

        for i in range( min(len(coef), len(const_coef)) ):
            c = coef[i]
            aberration += 2*np.pi / la * np.real(c * const_coef[i] * 
                omega**omega_power[i][0] * 
                omegabar**omega_power[i][1])
    elif basis == 'zernike':
        if conv_angle is None:
            raise ValueError('Convergence Angle is mandatory for basis `zernike`')
        rho, phi = ip.polarKernel(shape)
        rho /= conv_angle / px_size
        phi_func = [np.cos, np.sin]
        n, m, oddeven = [0, 0, 0]
        for c in coef:
            if (n-m)//2 != 0:
                m += 1
            
            ab = normzernike_r(n,m,rho)
            if m != 0:
                ab *= phi_func[oddeven](phi)
            aberration += ab

            oddeven += 1
            if (oddeven > 2) or (m==0):
                n, m, oddeven = [n, m+1, 0]
            if n<m:
                n, m = [n+1, 0]
    return np.exp(1j*aberration)
                
def depth_of_focus(angle, e0):
    #Born & Wolf, 1999; Cosgriff et al., 2008; Intaraprasonk et al., 2008.
    return 2*wavelength(e0)*1e-10/angle**2

