#%%
import os 
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import glob
import numpy as np
import h5py
import matplotlib.pyplot as plt

from tools import imageprocess as ip
from tools import physics
from phaseplate_dummy.phaseplate import PhasePlate
import tools.imagematcher as im
from tools.phaseretriever import GS

#%%
datadir_pp = '/mnt/B2C2DD54C2DD1E03/data/20230215/phaseplate_image/'

img_l = 2048
pp_image = ip.load(datadir_pp + '09_LM_92x.dm4')**2
pp_image[pp_image<0] = 0
hist, bin_edges = np.histogram(pp_image, bins=30)

thre = bin_edges[6]
pp_image[pp_image<thre] = 0
ip.show([pp_image!=0,pp_image])
np.save('phaseplate_dummy/pp_osiris',pp_image[760:1200,780:1200])



print('done')
