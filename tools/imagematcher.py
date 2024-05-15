import numpy as np
from scipy import ndimage

import tools.imageprocess as ip

def centering_com(img, ref, xy=None, mask=None):
    if xy is None:
        if mask is not None:
            com_img = ip.com(img*mask)
            com_ref = ip.com(ref*mask)
        else: 
            com_img = ip.com(img)
            com_ref = ip.com(ref)
        [x, y] = [com_ref[1] - com_img[1],
            com_ref[0] - com_img[0]]
    else:
        [x, y] = xy    
    img = ip.shift(img, x, y)
    return img, [x, y]



def centering_xcor(img, ref, xy=None, mask=None):
    L2 = ref.shape[0]//2        
    if xy is None:

        if mask is not None:
            map = ip.xcor(img*mask, ref*mask) 
        else: 
            map = ip.xcor(img, ref) 
        [x, y] = np.unravel_index(map.argmax(), map.shape)
    else:
        [x, y] = xy
    img = ip.shift(img, x-L2, y-L2)
    return img, [x, y]

def centering_xcor_fine(img, ref, xy = None):
    L2 = ref.shape[0]//2
    img_ori = np.copy(img)
    if xy is None:
        phase = np.exp(1j*np.angle(ip.fft2d(img)/ip.fft2d(ref)))
        pattern0 = np.abs(ip.fft2d(ref))>0.001
        pattern1 = ip.normalize(np.abs(ip.ifft2d(pattern0)), b_unitary=True)
        pattern2 = ip.normalize(np.abs(ip.ifft2d(phase*pattern0)), b_unitary=True)
        x1, y1 = (np.argmax(pattern1)//img.shape[0], np.argmax(pattern1)%img.shape[0])
        x2, y2 = (np.argmax(pattern2)//img.shape[0], np.argmax(pattern2)%img.shape[0])
        dx = x1-x2
        dy = y1-y2
        d=10
        x1, y1 = ip.com(pattern1[x1-d:x1+d+1, y1-d:y1+d+1])
        x2, y2 = ip.com(pattern2[x2-d:x2+d+1, y2-d:y2+d+1])
        img = ip.shift(img, (x1-x2)+dx, (y1-y2)+dy)
    else:
        [x, y] = xy
        img = ip.shift(img, x-L2, y-L2)
    return img, [x2-x1, y2-y1]

def mag_rad(img, ref, mag=None, mask=None): 

    if mag is None:
        rho_kernel, _ = ip.polarKernel(ref.shape)
        if mask is not None:
            mag = (ref*mask*rho_kernel).sum() / (img*mask*rho_kernel).sum()
        else:
            mag = (ref*rho_kernel).sum() / (img*rho_kernel).sum()

    img = ndimage.zoom(img, (mag,mag))
    # img = ip.rmBackground(img, ratio=0.01)
    # img = ip.normalize(img, b_unitary=True)
    img = ip.matchShape(img, ref.shape)
    return img, mag


def mag_rad_fine(img, ref, mag=None, mask=None):
    L = ref.shape[0]
    if mag is None:
        rho_kernel, _ = ip.polarKernel(ref.shape)
        if mask is not None:
            mag = (ref*mask*rho_kernel).sum() / (img*mask*rho_kernel).sum()
        else:
            mag = (ref*rho_kernel).sum() / (img*rho_kernel).sum()
        mag = np.min([mag, (L-1)/L])

    img = ndimage.zoom(img, (mag,mag))
    img = ip.matchShape(img, ref.shape)
    img[img<0] = 0
    return img, mag
