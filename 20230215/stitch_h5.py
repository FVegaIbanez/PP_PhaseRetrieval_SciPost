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

def copy(result, result_copy, range):
    result.iwph[range[0]:range[1]]   = result_copy.iwph[range[0]:range[1]]
    result.score[range[0]:range[1]]  = result_copy.score[range[0]:range[1]]
    result.init_p[range[0]:range[1]] = result_copy.init_p[range[0]:range[1]]
    return result

datadir_re = '/mnt/B2C2DD54C2DD1E03/data/20230215/recon_result/'

n_px = 48
n_step = 11
length = 2048

result = Result.create(datadir_re+'0223_searchphase.h5',n_px, n_step, length)

result_0 = Result.open(datadir_re+'0223_searchphase_0.h5')
result = copy(result, result_0, [0,12])
result_0.close()

result_1 = Result.open(datadir_re+'0223_searchphase_1.h5')
result = copy(result, result_1, [12,24])
result_1.close()

result_2 = Result.open(datadir_re+'0223_searchphase_2.h5')
result = copy(result, result_2, [24,36])
result_2.close()

result_3 = Result.open(datadir_re+'0223_searchphase_3.h5')
result = copy(result, result_3, [36,48])
result_3.close()

result.close()