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
    def __init__(self, filename, n_px, n_step, length):
        self.f = h5py.File(filename, 'r')
        self.iwph   = self.f.get("iwph")
    def close(self):
        self.f.close()

#%%
datadir_re = '/mnt/B2C2DD54C2DD1E03/data/20230215/recon_result/'
iw_data = np.load(datadir_re+'0216_iw.npy', allow_pickle=True).item()
iw = iw_data['pp_iw_wave']
idxmap = iw_data['idxmap']

n_px = 48
n_step = 11
length = 2048
result = Result(datadir_re+'0221_searchphase.h5',n_px, n_step, length)

px_list = [19,15,16,20,24,25,34,36,35,32,26,23,
        12,9,10,11,21,27,37,39,42,38,28,22,
        8,3,4,7,17,29,40,45,46,41,31,18,
        6,2,1,5,13,30,43,47,48,44,33,14
]



phase_map = np.zeros((n_px, n_step))


for ii_px in range(35,48):
    px_map = idxmap==px_list[ii_px]
    phase_0 = np.copy(result.iwph[ii_px, 0])[px_map]
    iw_px = iw[px_map]
    # phase_0 = 0

    for ii_step in range(n_step):
        phase = np.copy(result.iwph[ii_px, ii_step])[px_map]
        phase_d = phase-phase_0
        phase_val = np.angle( (iw_px*np.exp(1j * phase_d)).sum() )
        phase_map[ii_px, ii_step] = phase_val

ip.show(phase_map)
np.save(datadir_re + 'phasemap.npy', phase_map)

print('done')