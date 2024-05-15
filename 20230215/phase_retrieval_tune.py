#%%
import os 
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import numpy as np
import h5py
import matplotlib.pyplot as plt

from tools import imageprocess as ip
from phaseplate_dummy.phaseplate import PhasePlate
import tools.imagematcher as im
from tools.phaseretriever import GS

class Result():
    def __init__(self, filename):
        self.f = h5py.File(filename, 'a')
    @ classmethod
    def create(cls,filename, n_px, n_step, length):
        obj = cls(filename)
        obj.iwph   = obj.f.create_dataset("iwph", (n_px, n_step, length, length), dtype='single')
        # self.iwam   = self.f.create_dataset("iwam", (n_px, n_step, length, length), dtype='single', compression="gzip", compression_opts=9)
        # self.ewph   = self.f.create_dataset("ewph", (n_px, n_step, length, length), dtype='single', compression="gzip", compression_opts=9)
        # self.ewam   = self.f.create_dataset("ewam", (n_px, n_step, length, length), dtype='single', compression="gzip", compression_opts=9)
        # self.df     = self.f.create_dataset("df", (n_px, n_step, length, length), dtype='single', compression="gzip", compression_opts=9)
        # self.df     = self.f.get('df')
        obj.score  = obj.f.create_dataset("score", (n_px, n_step), dtype='single') 
        obj.b_vis  = obj.f.create_dataset("b_vis", (n_px, n_step), dtype='single') 
        obj.init_p = obj.f.create_dataset("init_p", (n_px, n_step), dtype='single') 
        return obj

    @ classmethod
    def open(cls,filename):
        obj = cls(filename)
        obj.iwph    = obj.f.get('iwph')
        obj.score   = obj.f.get('score')
        obj.b_vis   = obj.f.get('b_vis')
        obj.init_p  = obj.f.get('init_p')
        return obj

    def update(self, gs, i_px, i_step, p):
        if self.b_vis[i_px, i_step] == 0:
            self.iwph[i_px, i_step] = np.angle(gs.iw).astype('single')
            # self.iwam[i_px, i_step] = np.abs(gs.iw).astype('single')
            # self.ewph[i_px, i_step] = np.angle(gs.ew).astype('single')
            # self.ewam[i_px, i_step] = np.abs(gs.ew).astype('single')
            self.score[i_px, i_step] = gs.error[-1]
            self.b_vis[i_px, i_step] = 1
            self.init_p[i_px, i_step]= p
        elif self.score[i_px, i_step] > gs.error[-1]:
            self.iwph[i_px, i_step] = np.angle(gs.iw).astype('single')
            # self.iwam[i_px, i_step] = np.abs(gs.iw).astype('single')
            # self.ewph[i_px, i_step] = np.angle(gs.ew).astype('single')
            # self.ewam[i_px, i_step] = np.abs(gs.ew).astype('single')
            self.score[i_px, i_step] = gs.error[-1]
            self.init_p[i_px, i_step]= p
        print(gs.error[-1])

    def close(self):
        self.f.close()

#%%
datadir_re = '/mnt/B2C2DD54C2DD1E03/data/20230215/recon_result/'
datadir_df = '/mnt/B2C2DD54C2DD1E03/data/20230215/recon/'
filenames_df = ['Image_0006.dm4','Image_0138.dm4','Image_0270.dm4','Image_0402.dm4']

iw_data = np.load(datadir_re+'0216_iw.npy', allow_pickle=True).item()
f = h5py.File(datadir_re+'df_result.h5','r')
df_images_h5 = f.get('df')

df_phase = iw_data['prop']
def customized_propagator(wave, direction):    
    if direction == 1:
        return ip.ifft2d( wave * df_phase, b_norm=True )
    elif direction == -1:
        return ip.fft2d(wave, b_norm=True) / df_phase 

n_px = 48
n_step = 11
length = 2048
# result = Result.create(datadir_re+'0222_searchphase.h5',n_px, n_step, length)
result = Result.open(datadir_re+'0308.h5')

#%% run
px_list = [19,15,16,20,24,25,34,36,35,32,26,23,
        12,9,10,11,21,27,37,39,42,38,28,22,
        8,3,4,7,17,29,40,45,46,41,31,18,
        6,2,1,5,13,30,43,47,48,44,33,14
]
iw = iw_data['pp_iw_wave'].astype('single')
iw[iw<0] = 0
iw2 = iw**2

for i_px in range(n_px):

    df_image = np.array(df_images_h5[i_px][0]).astype('single')

    print(i_px)
    gs = GS(incident_wave=iw2, intensity=df_image, propergator=customized_propagator)
    gs.ph = result.iwph[i_px,1]
    gs.run(lim_diff=1e-9, lim_n=20)
    result.b_vis[i_px,0]=0
    result.update(gs, i_px, 0 , 0)


result.close()
print('done')


# np.save(datadir_re + '/result_0217.npy',result)

ip.show(result.iwph[0,0]-result.iwph[0,1])
ip.plot(result.init_p[0])
ip.show(result.score)

#%%

phase_map = np.zeros((n_px, n_step))


for ii_px in range(35):
    phase_0 = np.copy(result.iwph[ii_px, 0])
    # phase_0 = 0

    for ii_step in range(n_step):
        phase = np.copy(result.iwph[ii_px, ii_step])
        phase_d = phase-phase_0
        phase_val = np.angle( (iw_data['pp_iw_wave']*np.exp(1j * phase_d))[iw_data['idxmap']==px_list[ii_px]].sum() )
        phase_map[ii_px, ii_step] = phase_val

ip.show(phase_map)
