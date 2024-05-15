import numpy as np
import matplotlib.pyplot as plt
import imageprocess as ip
import time

class GS():

    def __init__(self, iw, ew, propergator):
        if isinstance(iw, list):
            self.iw_raw = [ _iw * ((abs(_ew)**2).sum() / (abs(_iw)**2).sum())  ** 0.5 for _iw, _ew in zip(iw, ew) ]
            self.ew_raw = [ (_ew) for _ew in ew ]
        else:
            self.iw_raw = [ iw* ((abs(ew)**2).sum() / (abs(iw)**2).sum()) ** 0.5 ]
            self.ew_raw = [ew]
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
        self.iw = [iw_raw * self._ph for iw_raw in self.iw_raw]
        self.ew = [np.exp(1j*np.angle(self.prop(self.obj*iw, 1)))*ew for iw, ew in zip(self.iw, self.ew_raw)] 

    def run(self, lim_n = 100):
        n = 0
        self.init_wave()
        self.error = np.zeros((len(self.iw), lim_n))
        # self.error.append(self.calcError(abs(self.prop(self.iw[0],1)), self.ew_raw[0]))

        start_time = time.time()
        while self.ite < (lim_n*len(self.iw)):
            self.gs()
            if self.b_probe and (self.ite%self.part_probe==0) and self.ite != 0:
                self.update_iw(rm_mean = False)
                # self.update_iw()
            self.ite += 1
        self.b_ref_one = False
        self.update_iw(rm_mean=False)
        # self.ew = [self.prop(iw*self.obj, 1) for iw in self.iw]
        print('\n GS over after ', self.ite, 'iteration.\n')
        print("--- %s seconds ---" % (time.time() - start_time))

    def gs(self):
        id = self.ite%len(self.iw)
        # self.iw[id] = np.exp( 1j * np.angle( self.prop(self.ew[id], -1) ) ) * self.iw_raw[id]
        new_iw = self.prop(self.ew[id], -1) * (self.iw_raw[id]!=0)
        if self.phase_obj:
            new_phase = new_iw[self.iw_raw[id]!=0]/self.iw[id][self.iw_raw[id]!=0]
            self.obj[self.iw_raw[id]!=0] *= \
                np.exp(1j * (self.update_step) * np.angle(new_phase / self.obj[self.iw_raw[id]!=0]))
        else:
            self.obj[self.iw_raw[id]!=0] += \
                (self.update_step) * new_iw[self.iw_raw[id]!=0]/self.iw[id][self.iw_raw[id]!=0] -\
                (self.update_step) * self.obj[self.iw_raw[id]!=0]
        self.ew[id] = self.prop(self.iw[id]*self.obj, 1)
        self.error[id, self.ite//len(self.iw)] = self.calcError(abs(self.ew[id]), self.ew_raw[id])
        self.ew[id] = np.exp( 1j * np.angle(self.ew[id]) ) * self.ew_raw[id]

    def update_iw(self, rm_mean=True):
        if self.b_ref_one:
            new_iw = self.prop(self.ew[0], -1) * (self.iw_raw[0]!=0)
            new_obj = new_iw[self.iw_raw[0]!=0]/self.iw[0][self.iw_raw[0]!=0]
            if self.phase_obj:
                self.obj[self.iw_raw[0]!=0] = \
                    np.exp(1j * np.angle(new_obj))
            else:
                self.obj[self.iw_raw[0]!=0] = new_obj

            self.iw = [self.iw[0]]
            for ew, iw_raw in zip(self.ew[1:], self.iw_raw[1:]):
                iw = np.copy(np.abs(iw_raw)).astype('complex')
                iw[iw_raw!=0] = np.exp(1j*np.angle(self.prop(ew,-1)[iw_raw!=0]/self.obj[iw_raw!=0])) * abs(iw_raw[iw_raw!=0])
                self.iw.append(iw)
        else:
            self.iw = []
            for ew, iw_raw in zip(self.ew, self.iw_raw):
                iw = np.copy(np.abs(iw_raw)).astype('complex')
                iw[iw_raw!=0] = np.exp(1j*np.angle(self.prop(ew,-1)[iw_raw!=0]/self.obj[iw_raw!=0])) * abs(iw_raw[iw_raw!=0])
                self.iw.append(iw)
        if rm_mean:
            if self.b_ref_one:
                mean_phase = np.angle(self.iw[1:]).mean(0)
            else:
                mean_phase = np.angle(self.iw).mean(0)
            mean_phase[self.mean_ignore] = 0
            self.iw = np.array(self.iw)
            self.iw[1:,self.iw_raw[0]!=0] *= np.exp(-1j*mean_phase[self.iw_raw[0]!=0])
            self.iw = list(self.iw)
            self.obj[self.iw_raw[0]!=0] *= np.exp(1j*mean_phase[self.iw_raw[0]!=0])
        

    @staticmethod
    def calcError(arr, gt_arr):
        return ((arr-gt_arr)**2).mean() / gt_arr.mean()






