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
datadir_re = '/mnt/B2C2DD54C2DD1E03/data/20230215/recon_result/'
datadir_df = '/mnt/B2C2DD54C2DD1E03/data/20230215/recon/'
filenames_df = ['Image_0006.dm4','Image_0138.dm4','Image_0270.dm4','Image_0402.dm4']

iw_data = np.load(datadir_re+'0216_iw.npy', allow_pickle=True).item()

df_phase = iw_data['prop']
def customized_propagator(wave, direction):    
    if direction == 1:
        return ip.ifft2d( wave * df_phase, b_norm=True )
    elif direction == -1:
        return ip.fft2d(wave, b_norm=True) / df_phase 


images_df = []
for name in filenames_df:
    image = ip.load(datadir_df + name)
    image[image<0] = 0
    image,_ = im.centering_com(image, np.ones(image.shape))
    image[image<0] = 0
    image,_ = im.centering_com(image, np.ones(image.shape))
    image[image<0] = 0
    image = ip.normalize(image, b_unitary=True)
    images_df.append(image)

px_list = [19,12,8,2]
images_pp = []
images_ppiw = []
for image_df, px in zip(images_df,px_list):
    image_pp = iw_data['pp_image']
    image_ppiw = abs(iw_data['iw'])

    gs = GS(incident_wave=image_pp**2, intensity=image_df, propergator=customized_propagator)
    gs.ph = np.angle(iw_data['iw'])
    gs.run(lim_diff=1e-9, lim_n=100)
    images_pp.append(gs.iw)

    gs = GS(incident_wave=image_ppiw**2, intensity=image_df, propergator=customized_propagator)
    gs.ph = np.angle(iw_data['iw'])
    gs.run(lim_diff=1e-9, lim_n=100)
    images_ppiw.append(gs.iw)

fig, axes = plt.subplots(3, 4)
for i_x in range(4):
    axes[0,i_x].imshow(images_df[i_x])
    axes[1,i_x].imshow((np.angle(images_pp[i_x] / iw_data['iw'])*(iw_data['idxmap_sharp']))[500:-500,500:-500])  
    axes[2,i_x].imshow((np.angle(images_ppiw[i_x] / iw_data['iw'])*(iw_data['idxmap_sharp']))[500:-500,500:-500])
plt.show(block=False)

print('done')
