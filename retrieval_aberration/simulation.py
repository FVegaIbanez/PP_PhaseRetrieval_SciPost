import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append('/mnt/B2C2DD54C2DD1E03/softwares/python_modules/')

from tools.phaseretriever import GS
import tools.imageprocess as ip
import numpy as np
import matplotlib.pyplot as plt
from phaseplate_dummy.phaseplate import PhasePlate
import basis

def prop(wave, direction):    
    if direction == 1:
        return ip.ifft2d( wave, b_norm=True )
    elif direction == -1:
        return ip.fft2d(wave, b_norm=True)
    
def apply_phase(wave, phase_list, idx_list):
    wave = wave.astype('complex')
    phase_list = np.array(phase_list)
    phase_list = np.exp(1j*phase_list)
    for phase, idx in zip(phase_list, idx_list):
        wave[idx] *= phase
    return wave

basis = basis.make_basis_f()

image_length = 512
fill_ratio = 0.2
rot_angle = -np.pi*0.11 + 2*np.pi/12 * 0

pp = PhasePlate(np.load('phaseplate_dummy/pp_osiris.npy')) 
pp.setFrame(image_length, fill_ratio, rot_angle)
pp.indexPixels()

iw_amp_0 = np.copy(pp.wave)
idxmp = pp.idx_map
mplist = pp.map_list

picks = [31, 25, 18, 2, 40, 7, 13, 22, 45, 37]
aberration = np.exp(1j*ip.normalize(ip.frqPatch(iw_amp_0.shape, 0.05, 0, mode='gauss-lowpass'), (0,np.pi*2)))
phase_uncertainty = (np.random.rand(len(picks),48)*2-1)*0.2
px_phase_list = [[v + p for v, p in zip(vec, pu)] for vec, pu in zip(basis[picks], phase_uncertainty)]
iw = [apply_phase(iw_amp_0, phase, mplist) for phase in px_phase_list]
iw_exp = [apply_phase(iw_amp_0, phase, mplist) for phase in basis[picks]]
ew_amp = [abs(prop(wave*aberration,1)) for wave in iw]

## run ########################
gs = GS(iw_exp, ew_amp, prop)
gs.b_probe = True
# gs.b_ref_one = True
gs.beg_probe = 30
gs.part_probe = 1
gs.phase_obj = True
# gs.obj = np.copy(aberration)
gs.run(lim_n = 50)
###############################

phase = np.angle((aberration/gs.obj))
phase[phase<0] += np.pi*2
phase_mean = phase[idxmp!=0].mean()
phase[idxmp!=0]-=phase_mean

plt.figure()
plt.imshow(phase*(idxmp!=0), vmin = -np.pi/10, vmax = np.pi/10)
plt.show(block=False)
error = np.sqrt((phase[idxmp!=0]**2).mean())
print(error)

iw_phase = (np.angle(iw[0]*aberration)-np.angle(gs.iw[0]*gs.obj))*(idxmp!=0)
iw_phase[iw_phase<0] += np.pi*2
iw_phase[idxmp!=0] -= iw_phase[idxmp!=0].mean()
ip.show(iw_phase)

print('done')













