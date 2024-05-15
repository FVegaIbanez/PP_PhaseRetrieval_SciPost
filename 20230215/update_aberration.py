#%%
import os 
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append('/mnt/B2C2DD54C2DD1E03/softwares/python_modules/')

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
        obj.ab     = obj.f.create_dataset("ab", (n_px, length, length), dtype='single') 
        return obj

    @ classmethod
    def open(cls,filename):
        obj = cls(filename)
        obj.iwph    = obj.f.get('iwph')
        obj.score   = obj.f.get('score')
        obj.b_vis   = obj.f.get('b_vis')
        obj.init_p  = obj.f.get('init_p')
        obj.ab      = obj.f.get('ab') 
        obj.b_vis   = np.zeros(obj.b_vis.shape)
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
    return iw_set


#%%4
datadir_re = '/mnt/B2C2DD54C2DD1E03/data/20230215/recon_result/'
datadir_df = '/mnt/B2C2DD54C2DD1E03/data/20230215/recon/'
filenames_df = ['Image_0006.dm4','Image_0138.dm4','Image_0270.dm4','Image_0402.dm4']

iw_data = np.load(datadir_re+'0216_iw.npy', allow_pickle=True).item()
f = h5py.File(datadir_re+'0315_df_result.h5','r')
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
# result = Result.create(datadir_re+'test.h5',n_px, n_step, length)
# result = Result.open(datadir_re+'test.h5')
# result_old = Result.open(datadir_re+'0308.h5')

#%% run
px_list = [19,15,16,20,24,25,34,36,35,32,26,23,
           12, 9,10,11,21,27,37,39,42,38,28,22,
            8, 3, 4, 7,17,29,40,45,46,41,31,18,
            6, 2, 1, 5,13,30,43,47,48,44,33,14
]
iw = iw_data['pp_iw_wave'].astype('single')
idxmp = iw_data['idxmap']
iw[idxmp==0] = 0

# df_set = list(np.array(df_images_h5[:, 5]).astype('single')**0.5)
# iw_pi_list = []
# for i_px in range(n_px):
#     iw_pi = np.copy(iw)
#     iw_pi[idxmp == px_list[i_px]] *= -1
#     iw_pi_list.append(iw_pi)
# gs = GS(iw_pi_list, df_set, customized_propagator)
# gs.b_ref_one = False
# gs.b_probe = True
# gs.part_probe = 48*2
# gs.phase_obj = True
# gs.run(lim_n = 6)
# obj_phase = np.angle(gs.obj)
# np.save(datadir_re + '0315_ab.npy', obj_phase)

obj_phase = np.load(datadir_re + '0315_ab_updated.npy')
obj_cplx = np.exp(1j * obj_phase)

# df_set = list(np.array(df_images_h5[:, 0]).astype('single')**0.5)
# n_try = 15
# for i_try in range(n_try):
#     for i_px in range(n_px):

#         # plus
#         obj_test = np.copy(obj_cplx)
#         obj_test[idxmp==(i_px+1)] *= np.exp(1j * np.pi / (1.3**i_try) )

#         gs = GS(iw, df_set[i_px], customized_propagator)
#         score0 = gs.calcError(abs(gs.prop(gs.iw[0]*obj_cplx,1)), gs.ew_raw[0])
#         score1 = gs.calcError(abs(gs.prop(gs.iw[0]*obj_test,1)), gs.ew_raw[0])

#         if score1 < score0:
#             obj_phase = np.angle(obj_test)
#             obj_cplx = obj_test

#         # minus
#         obj_test = np.copy(obj_cplx)
#         obj_test[idxmp==(i_px+1)] *= np.exp(-1j * np.pi / (2**i_try) )

#         gs = GS(iw, df_set[i_px], customized_propagator)
#         score0 = gs.calcError(abs(gs.prop(gs.iw[0]*obj_cplx,1)), gs.ew_raw[0])
#         score1 = gs.calcError(abs(gs.prop(gs.iw[0]*obj_test,1)), gs.ew_raw[0])

#         if score1 < score0:
#             obj_phase = np.angle(obj_test)
#             obj_cplx = obj_test

#         print(i_try, i_px, score0, score1)
        

obj_phase_2 = np.copy(obj_phase)
df_set_0 = list(np.array(df_images_h5[:,0]).astype('single')**0.5)
df_set_5 = list(np.array(df_images_h5[:,5]).astype('single')**0.5)
init_phase = np.ones(n_px)
offset = np.zeros(n_px)


ite = 0
while (init_phase!=0).any():
    print(ite)
    ite+=1
    init_phase = np.zeros(n_px)
    for i_px in range(n_px):

        _iw = np.copy(iw).astype('complex')
        ph = np.copy(obj_phase_2)
        ph[idxmp == px_list[i_px]] += np.pi - offset[i_px]

        _iw *= np.exp(1j*ph)

        gs = GS([_iw], [df_set_5[i_px]], customized_propagator)
        gs.phase_obj = True
        gs.update_step = 1
        gs.run(lim_n = 1)
        obj_phase_3 = np.angle(gs.obj) + obj_phase_2

        score = []
        for p in np.arange(-np.pi, np.pi, np.pi/10):
            _iw = np.copy(iw).astype('complex')
            ph = np.copy(obj_phase_3)
            ph[idxmp == px_list[i_px]] += p

            _iw *= np.exp(1j*ph)

            gs = GS(_iw, df_set_0[i_px], customized_propagator)

            score.append(gs.calcError(abs(gs.prop(gs.iw[0],1)), gs.ew_raw[0]))

        score = np.array(score)
        init_phase[i_px] = np.arange(-np.pi, np.pi, np.pi/10)[score.argmin()]
        offset[i_px] += np.arange(-np.pi, np.pi, np.pi/10)[score.argmin()]


        # obj_phase_2[idxmp==px_list[i_px]] += init_phase[i_px]



np.save(datadir_re + '0315_ab_updated.npy', obj_phase)









print('done')






# ip.show(result.init_p)


fixed = [8,17,19,32,33]