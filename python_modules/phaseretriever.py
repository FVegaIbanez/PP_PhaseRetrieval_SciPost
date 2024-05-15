import numpy as np
import matplotlib.pyplot as plt
import imageprocess as ip
import time
import torch

class GS():

    def __init__(self, iw, ew, propergator, gpu=False):
        if len(iw.shape)==3:
            self.iw_raw = iw * ((abs(ew)**2).sum((-1,-2)) / (abs(iw)**2).sum((-1,-2)))  ** 0.5
        else:
            iw = torch.tensor(iw)
            ew = torch.tensor(ew)
            self.iw_raw = torch.unsqueeze(iw* ((abs(ew)**2).sum() / (abs(iw)**2).sum()) ** 0.5, 0)
            ew = torch.unsqueeze(ew, 0)
        self.ew_raw = ew
        self.n0 = self.iw_raw!=0
        self.prop = propergator
        self._ph = np.ones(self.iw_raw[0].shape)
        self.ite = 0
        
        self.obj = np.ones(self.iw_raw[0].shape,dtype='complex')
        self.b_probe = False
        self.part_probe = 1
        self.beg_probe = 0
        self.phase_obj = False
        self.update_step = 0.3
        self.b_ref_one = False
        self.gpu=gpu

    @property
    def ph(self):
        return self._ph

    @ph.setter
    def ph(self, p):
        if not np.iscomplex(p).any():
            p = np.exp(1j*p)
        self._ph = p

    def init_wave(self):
        self.iw_raw = torch.tensor(self.iw_raw)
        self.ew_raw = torch.tensor(self.ew_raw)
        self.obj = torch.tensor(self.obj)
        self.iw_raw = self.iw_raw.type(torch.complex64)
        self.ew_raw = self.ew_raw.type(torch.float32)
        self.obj = self.obj.type(torch.complex64)
        self.iw = torch.clone(self.iw_raw)
        if self.gpu:
            self.obj = self.obj.to("cuda")
            self.ew_raw = self.ew_raw.to("cuda")
            self.iw_raw = self.iw_raw.to("cuda")
            self.iw = self.iw.to("cuda")
        # self.iw = torch.tensor(self.iw_raw * self._ph).to("cuda")
        self.iw_temp = torch.clone(self.iw_raw)
        self.obj_temp = torch.clone(self.obj)
        self.ew = self.prop(self.obj*self.iw, 1)
        self.ew *= torch.abs(self.ew_raw) / torch.abs(self.ew)

    def fetch(self):
        if self.gpu:
            self.iw = self.iw.clone().detach().to('cpu')
            self.ew = self.ew.clone().detach().to('cpu')
            self.iw_raw = self.iw_raw.clone().detach().to('cpu')
            self.ew_raw = self.ew_raw.clone().detach().to('cpu')
            self.obj = self.obj.clone().detach().to('cpu')

    def run(self, lim_n = 100):
        n = 0
        self.init_wave()
        self.error = torch.zeros((len(self.iw), lim_n))
        start_time = time.time()
        while self.ite < lim_n:
            self.gs()
            if self.b_probe and (self.ite%self.part_probe==0) and self.ite > self.beg_probe:
                # self.update_iw(rm_mean = False)
                self.update_iw()
            self.update_ew()
            self.ite += 1
        self.b_ref_one = False
        # self.update_iw(rm_mean=False)
        self.gs()
        self.update_iw()
        self.fetch()
        print('\n GS over after ', self.ite, 'iteration.\n')
        print("--- %s seconds ---" % (time.time() - start_time))

    def gs(self):
        self.iw_temp = self.prop(self.ew, -1)
        self.obj[self.n0[0]] = (self.iw_temp[:,self.n0[0]] / self.iw[:,self.n0[0]]).mean(0)
        if self.phase_obj:
            self.obj[self.n0[0]] = self.obj[self.n0[0]] / torch.abs(self.obj[self.n0[0]]) 
        

        # for id in range(len(self.iw)):
        #     idx = self.n0[id]
        #     if self.phase_obj:
        #         self.obj[idx] *= \
        #             torch.exp(1j * (self.update_step) * torch.angle(self.iw_temp[id][idx] / self.iw[id][idx] / self.obj[idx]))
        #     else:
        #         self.obj[idx] += \
        #             (self.update_step) * self.iw_temp[id][idx]/self.iw[id][idx] -\
        #             (self.update_step) * self.obj[idx]
        #     self.ew[id] = self.prop(self.iw[id]*self.obj, 1)
        #     self.error[id, self.ite] = self.calcError(abs(self.ew[id]), self.ew_raw[id])
        # self.ew = self.ew / torch.abs(self.ew) * self.ew_raw

    def update_iw(self, rm_mean=True):
        self.iw_temp = self.prop(self.ew, -1) 
        self.iw_temp[:,self.n0[0]] /= self.obj[self.n0[0]]
        self.iw_temp = torch.angle(self.iw_temp)
        for id in range(len(self.iw)):
            # self.iw[id][self.n0[id]] = self.iw_temp[id][self.n0[id]]/self.obj[self.n0[id]]*torch.abs(self.obj[self.n0[id]])
            self.iw[id][self.n0[id]] = torch.exp(1j*
                self.iw_temp[id][self.n0[id]]*self.update_step - torch.angle(self.iw[id][self.n0[id]])*(1-self.update_step)                                      
            )*torch.abs(self.iw_raw[id][self.n0[id]]) 


        idx = self.n0[0]
        if self.b_ref_one:
            self.iw[:,idx] *= torch.abs(self.iw[0][idx]) / self.iw[0][idx]
            self.obj[idx] *= self.iw[0][idx] / torch.abs(self.iw[0][idx])

    def update_ew(self):
        self.ew = self.prop(self.iw*self.obj,1)
        for id in range(len(self.ew)):
            self.error[id, self.ite] = self.calcError(abs(self.ew[id]), self.ew_raw[id])
        self.ew = self.ew / torch.abs(self.ew) * self.ew_raw

        

    @staticmethod
    def calcError(arr, gt_arr):
        return ((arr-gt_arr)**2).mean() / gt_arr.mean()






