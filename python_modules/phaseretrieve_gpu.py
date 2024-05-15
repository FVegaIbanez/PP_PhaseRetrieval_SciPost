import numpy as np
import matplotlib.pyplot as plt
import imageprocess as ip
import time

import torch 

class GS():

    def __init__(self, iw, ew, propergator):
        self.init_gpu()
        if isinstance(iw, list):
            self.iw_raw = np.array([ _iw * ((abs(_ew)**2).sum() / (abs(_iw)**2).sum())  ** 0.5 for _iw, _ew in zip(iw, ew) ])
            self.ew_raw = np.array([ (_ew) for _ew in ew ])
        else:
            self.iw_raw = np.array([ iw* ((abs(ew)**2).sum() / (abs(iw)**2).sum()) ** 0.5 ])
            self.ew_raw = np.array([ew])
        self.obj = np.ones(self.iw_raw[0].shape,dtype='complex')
        self.prop = propergator
        self._ph = np.ones(self.iw_raw[0].shape)
        self.iw = self.iw_raw
        self.ew = self.ew_raw
        self.pp = None
        self.mask = None
        self.region_dc = []
        self.ite = 0
        self.b_probe = False
        self.part_probe = 1
        self.beg_probe = 0
        self.phase_obj = False
        self.update_step = 0.3
        self.b_ref_one = False
        self.mean_ignore = np.zeros(self.iw[0].shape, dtype='bool')
       

    @property
    def ph(self):
        return self._ph

    @ph.setter
    def ph(self, p):
        if not np.iscomplex(p).any():
            p = np.exp(1j*p)
        self._ph = p

    def init_wave(self):
        self.iw = self.iw_raw * self._ph
        self.ew = np.exp(1j*np.angle(self.prop(self.obj*self.iw, 1)))*self.ew_raw



    def run(self, lim_n = 100):
        n = 0
        self.init_wave()
        self.move_to_gpu()
        self.error = np.zeros((len(self.iw), lim_n))
        # self.error.append(self.calcError(abs(self.prop(self.iw[0],1)), self.ew_raw[0]))

        start_time = time.time()
        while self.ite < lim_n:

            self.gs()
            if self.b_probe and (self.ite%self.part_probe==0) and self.ite > self.beg_probe:
                # self.update_iw(rm_mean = False)
                self.update_iw()
            self.ite += 1
        self.b_ref_one = False
        self.update_iw(rm_mean=False)
        # self.update_iw()
        print('\n GS over after ', self.ite, 'iteration.\n')
        print("--- %s seconds ---" % (time.time() - start_time))

    def gs(self):
        new_iw = self.prop(self.ew, -1) * (self.iw_raw!=0)
        for id in range(len(self.iw)):
            idx = self.iw_raw[id]!=0
            if self.phase_obj:
                new_phase = new_iw[id][idx]/self.iw[id][idx]
                self.obj[idx] *= \
                    np.exp(1j * (self.update_step) * np.angle(new_phase / self.obj[idx]))
            else:
                self.obj[idx] += \
                    (self.update_step) * new_iw[id][idx]/self.iw[id][idx] -\
                    (self.update_step) * self.obj[idx]
            self.ew[id] = self.prop(self.iw[id]*self.obj, 1)
            self.error[id, self.ite] = self.calcError(abs(self.ew[id]), self.ew_raw[id])
            self.ew[id] = np.exp( 1j * np.angle(self.ew[id]) ) * self.ew_raw[id]

    def update_iw(self, rm_mean=True):
        iw_copy = np.copy(self.iw)
        if self.b_ref_one:
            new_iw = self.prop(self.ew[0], -1) * (self.iw_raw[0]!=0)
            new_obj = new_iw[self.iw_raw[0]!=0]/self.iw[0][self.iw_raw[0]!=0]
            if self.phase_obj:
                self.obj[self.iw_raw[0]!=0] = \
                    np.exp(1j * np.angle(new_obj))
            else:
                self.obj[self.iw_raw[0]!=0] = new_obj
            iw_0 = np.copy(self.iw[0])
            iw = self.prop(self.ew,-1)
            idx = iw!=0
            iw[idx] = np.abs(iw_copy[idx]) * np.exp(1j*np.angle(iw[idx]/np.tile(self.obj,(len(iw),1,1))[idx]))
            self.iw = iw
            self.iw[0] = iw_0
        else:
            iw = self.prop(self.ew,-1)
            idx = iw!=0
            iw[idx] = np.abs(iw_copy[idx]) * np.exp(1j*np.angle(iw[idx]/np.tile(self.obj,(len(iw),1,1))[idx]))
            self.iw = iw
        if rm_mean:
            idx = self.iw_raw[0]!=0
            # iw_diff = self.iw[:,idx] / iw_copy[:,idx]
            iw_diff = self.iw[:,idx] / self.iw_raw[:,idx]
            if self.b_ref_one:
                mean_phase = np.angle(iw_diff[1:].mean(0))
                self.iw[1:,idx] *= np.exp(-1j*mean_phase)
            else:
                mean_phase = np.angle(iw_diff.mean(0))
                self.iw[:,idx] *= np.exp(-1j*mean_phase)
            self.obj[idx] *= np.exp(1j*mean_phase)
        

    @staticmethod
    def calcError(arr, gt_arr):
        return ((arr-gt_arr)**2).mean() / gt_arr.mean()






