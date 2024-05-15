import numpy as np
import matplotlib.pyplot as plt
import tools.imageprocess as ip
from scipy import ndimage
import phaseplate_dummy as pp
from numba import jit
import time

class GS():

    def __init__(self, incident_wave, intensity, propergator):
        self.iw_raw = (incident_wave / incident_wave.sum() * intensity.sum()) ** 0.5 
        self.ew_raw = (intensity) ** 0.5 
        self.prop = propergator
        self._ph = np.ones(self.iw_raw.shape)
        self.iw = self.iw_raw * self._ph
        self.ew = np.copy(self.ew_raw)
        self.pp = None
        self.mask = None
        self.region_dc = []

    @property
    def ph(self):
        return self._ph

    @ph.setter
    def ph(self, p):
        if not np.iscomplex(p).any():
            p = np.exp(1j*p)
        self._ph = p

    def init_wave(self):
        iw = self.iw_raw*self._ph
        return iw, np.exp(1j*np.angle(self.prop(iw, 1))) * self.ew_raw

    def run(self, lim_n = 100, lim_diff = 1e-2):
        n = 0
        diff = np.inf   
        self.iw, self.ew = self.init_wave()
        self.error = []
        self.error.append( self.calcError(abs(self.prop(self.iw,1)), self.ew_raw) )

        start_time = time.time()
        # while n < lim_n and (diff > lim_diff):
        while n < lim_n :
            if self.mask is None:
                self.gs()
            else:
                self.gs_mask()
            # self.gs_fixDC()
            # self.gs_greedy(n)
            n += 1
        self.ew = self.prop(self.iw, 1)
        _ew = np.exp( 1j * np.angle(self.ew) ) * self.ew_raw
        self.iw = self.prop(_ew, -1)
        self.error = np.array(self.error)
        print('\n GS over after ', n, 'iteration.\n')
        print("--- %s seconds ---" % (time.time() - start_time))

    def gs(self):
        self.iw = np.exp( 1j * np.angle( self.prop(self.ew, -1) ) ) * self.iw_raw
        self.ew = self.prop(self.iw, 1)
        self.error.append( self.calcError(abs(self.ew), self.ew_raw) )
        self.ew = np.exp( 1j * np.angle(self.ew) ) * self.ew_raw

    # def gs_mask(self):
    #     self.ew = np.exp( 1j * np.angle(self.ew) ) * self.ew_raw
    #     iw_0 = np.exp( 1j * np.angle( self.prop(self.ew, -1) ) ) * self.iw_raw
    #     iw_m = iw_0 * np.exp(-1j * self.mask * np.pi/10)
    #     iw_p = iw_0 * np.exp(1j * self.mask * np.pi/10)
    #     ew_0 = self.prop(iw_0, 1)
    #     ew_m = self.prop(iw_m, 1)
    #     ew_p = self.prop(iw_p, 1)
    #     mini = np.argmin([self.calcError(abs(ew_0), self.ew_raw),
    #         self.calcError(abs(ew_m), self.ew_raw),
    #         self.calcError(abs(ew_p), self.ew_raw)])
    #     self.ew = [ew_0, ew_m, ew_p][mini]
    #     self.iw = [iw_0, iw_m, iw_p][mini]


    def gs_mask(self):
        _iw = self.prop(self.ew, -1)
        phase = np.where(self.mask, np.angle( _iw ), np.angle(self.iw))
        self.iw = np.exp( 1j * phase ) * self.iw_raw
        self.ew = self.prop(self.iw, 1)
        self.error.append( self.calcError(abs(self.ew), self.ew_raw) )
 

    def gs_fixDC(self):
        self.ew = np.exp( 1j * np.angle(self.ew) ) * self.ew_raw
        self.iw = np.exp( 1j * np.angle( self.prop(self.ew, -1) ) ) * self.iw_raw
        for re in self.region_dc:
            self.iw[re] = self.iw[re] / np.exp(1j * np.angle(self.iw[re].sum()))
        self.ew = self.prop(self.iw, 1)

    def gs_greedy(self, n):
        n_fails = 0
        if n!= 0:
            er = np.array(self.error)
            n_fails = (er[1:]>er[:-1]).sum()
        diff_ew = (self.ew_raw - np.abs(self.ew)) * (1 + 1 / (1+n_fails)) + np.abs(self.ew)
        # diff_ew = (self.ew_raw)
        self.ew = np.exp( 1j * np.angle(self.ew) ) * diff_ew
        # diff_iw = (np.angle(self.prop(self.ew, -1)) - np.angle(self.iw)) * (1 + 1 / (2+n_fails)) + np.angle(self.iw)
        diff_iw = (np.angle(self.prop(self.ew, -1)))
        self.iw = np.exp( 1j * diff_iw ) * self.iw_raw
        self.ew = self.prop(self.iw, 1)

    @staticmethod
    def calcError(arr, gt_arr):
        return ((arr-gt_arr)**2).mean() / gt_arr.mean()

