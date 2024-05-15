#%%
import os 
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import glob
import numpy as np
import h5py
import matplotlib.pyplot as plt

from tools import imageprocess as ip
from tools import physics
from phaseplate_dummy.phaseplate import PhasePlate
import tools.imagematcher as im
from tools.phaseretriever import GS




#%%
datadir_pp = '/mnt/B2C2DD54C2DD1E03/data/20230215/phaseplate_image/'
datadir_df = '/mnt/B2C2DD54C2DD1E03/data/20230215/recon/'

img_l = 2048
pp = PhasePlate(np.load('phaseplate_dummy/pp_osiris.npy')) # for ring0
fill_ratio = 0.195
pp.setFrame(img_l, fill_ratio, -np.pi*0.11 + 2*np.pi/12 * 0)
# pp.assignProperty(12, 300, mode='angle')
pp.assignProperty(0.18*1e-3, 300, mode='angle')

pp.indexPixels()
idxmap_sharp = pp.idx_map
cutoff = pp.wave.shape[0] * 0.1
rho,_ = ip.polarKernel(pp.wave.shape)
gauss = ip.normalize(1/(2*np.pi)**2/cutoff * np.exp(-rho**2/2/cutoff**2))
pp.wave = np.real(ip.ifft2d( ip.fft2d(pp.wave) * gauss ))
pp.wave[pp.wave<0]=0

# df_value = 0.73e4
df_value = 32444444444444.445
# df_value = 35526666666666.67
c3_value = 0.0e-3
tilt = (0,0)

tx, ty = ip.cartKernel(pp.wave.shape)
tilt_phase = np.exp(1j * (tx*tilt[0] + ty*tilt[1]) * 2*np.pi / pp.wave.shape[0])
df_phase = physics.aberration(pp.E0, pp.mrad_per_px, pp.wave.shape, [df_value * 1e-10, c3_value, 0, 0]) * tilt_phase
def customized_propagator(wave, direction):    
    if direction == 1:
        return ip.ifft2d( wave * df_phase, b_norm=True )
    elif direction == -1:
        return ip.fft2d(wave, b_norm=True) / df_phase 

image_df = ip.load(datadir_df + 'Image_0265.dm4')
image_df[image_df<0] = 0
image_df = image_df**1.4
image_df = ip.normalize(image_df, b_unitary=True)
image_pp = ip.normalize(abs( customized_propagator(pp.wave, 1) )**2, b_unitary=True)
image_df,_ = im.centering_com(image_df, image_pp)
image_df,_ = im.centering_com(image_df, image_pp)

image_df = ip.normalize(image_df,(0,1.8))
image_pp = ip.normalize(image_pp)

ip.show([image_df, image_pp, image_df-image_pp])
print(((image_df-image_pp)**2).sum()**0.5)



data = {
    'pp_iw_wave': pp.wave,
    'pp_df_int': image_pp,
    'prop': df_phase,
    'idxmap': pp.idx_map,
    'idxmap_sharp':idxmap_sharp
}

np.save('/mnt/B2C2DD54C2DD1E03/data/20230215/recon_result/0216_iw.npy', data)
