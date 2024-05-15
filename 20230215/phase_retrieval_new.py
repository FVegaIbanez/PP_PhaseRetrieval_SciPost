#%%
import os 
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append('/mnt/B2C2DD54C2DD1E03/softwares/python_modules/')

import numpy as np
import h5py
import matplotlib.pyplot as plt

from tools.phaseretriever import GS
import tools.imageprocess as ip

GPU = True

if GPU:
    import torch

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
        obj.ab     = obj.f.create_dataset("ab", (n_px, length, length), dtype='single') 
        return obj

    @ classmethod
    def open(cls,filename):
        obj = cls(filename)
        obj.iwph    = obj.f.get('iwph')
        obj.score   = obj.f.get('score')
        obj.b_vis   = obj.f.get('b_vis')
        obj.init_p  = obj.f.get('init_p')
        obj.init_ab = obj.f.get('ab') 
        obj.ab      = obj.f.get('ab_new') 

        # obj.b_vis = np.zeros(obj.b_vis.shape)
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

def make_iw_set(iw, init_phase, idxmp, i_px):
    iw_set = []
    for p in init_phase:
        _iw = np.copy(iw).astype('complex')
        _iw[idxmp==i_px] *= np.exp(1j*p)
        iw_set.append(_iw)
    return np.array(iw_set)


#%%
datadir_re = '/mnt/B2C2DD54C2DD1E03/data/20230215/recon_result/'
datadir_df = '/mnt/B2C2DD54C2DD1E03/data/20230215/recon/'
filenames_df = ['Image_0006.dm4','Image_0138.dm4','Image_0270.dm4','Image_0402.dm4']

iw_data = np.load(datadir_re+'0216_iw.npy', allow_pickle=True).item()
f = h5py.File(datadir_re+'0315_df_result.h5','r')
df_images_h5 = f.get('df')

if GPU:
    df_phase = torch.tensor(iw_data['prop'].astype('complex64')).to("cuda")
    def customized_propagator(wave, direction):
        if direction == 1:
            return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(
                wave * df_phase
            ))) * (wave.shape[-2] * wave.shape[-1])**0.5
        elif direction == -1:
            return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(
                wave
            ))) / (wave.shape[-2] * wave.shape[-1])**0.5 / df_phase

else:
    df_phase = iw_data['prop'].astype('complex64')
    def customized_propagator(wave, direction):    
        if direction == 1:
            return ip.ifft2d( wave * df_phase, b_norm=True )
        elif direction == -1:
            return ip.fft2d(wave, b_norm=True) / df_phase 

n_px = 48
n_step = 11
length = 2048
# result = Result.create(datadir_re+'0315.h5',n_px, n_step, length)
result = Result.open(datadir_re+'0402.h5')
# result_old = Result.open(datadir_re+'0315.h5')

#%% run
px_list = [19,15,16,20,24,25,34,36,35,32,26,23,
        12,9,10,11,21,27,37,39,42,38,28,22,
        8,3,4,7,17,29,40,45,46,41,31,18,
        6,2,1,5,13,30,43,47,48,44,33,14
]
iw = iw_data['pp_iw_wave'].astype('single')
idxmp = iw_data['idxmap']
iw[idxmp==0] = 0
iw2 = iw**2


for i_px in range(n_px):
# for i_px in [17]:
    print(i_px)

    df_set = np.array(df_images_h5[i_px]).astype('single')**0.5

    init_phase = result.init_p[i_px]
    iw_set = make_iw_set(iw, init_phase, idxmp, px_list[i_px])
    gs = GS(iw_set, df_set, customized_propagator, GPU)
    gs.b_ref_one = True
    gs.b_probe = True
    gs.part_probe = 20
    gs.beg_probe = 20
    gs.phase_obj = True
    gs.obj = np.exp(1j*result.init_ab[i_px])
    gs.mean_region = (idxmp!=px_list[i_px]) & (idxmp!=0)
    gs.run(lim_n = 750)

    result.iwph[i_px] = np.angle(np.array(gs.iw))
    result.ab[i_px] = np.angle(gs.obj)

# ip.show(np.angle(gs.iw[0])*(idxmp!=0))


result.close()
print('done')

# torch.tensor([0.0176, 0.0178, 0.0176, 0.0182, 0.0170, 0.0176, 0.0169, 0.0174, 0.0181,
#         0.0170, 0.0172])
# torch.tensor([0.0170, 0.0172, 0.0182, 0.0178, 0.0161, 0.0178, 0.0169, 0.0173, 0.0172,
#         0.0162, 0.0177])