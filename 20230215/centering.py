#%%
import os 
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import numpy as np
import h5py
import matplotlib.pyplot as plt

from tools import imageprocess as ip
import tools.imagematcher as im


datadir_re = '/mnt/B2C2DD54C2DD1E03/data/20230215/recon_result/'
datadir_df = '/mnt/B2C2DD54C2DD1E03/data/20230215/recon/'

iw_data = np.load(datadir_re+'0216_iw.npy', allow_pickle=True).item()

n_px = 48
n_step = 11
length = 2048

with h5py.File(datadir_re + '0315_df_result.h5', 'w') as f:
    ds = f.create_dataset('df', (n_px, n_step, length, length), dtype='single')

    for i_px in range(n_px):

        for i_step in range(n_step):

            id = i_px * n_step + i_step        
            name = 'Image_' + (4-len(str(id+1))) * '0' + str(id+1) + '.dm4'
            print(i_px, i_step, name)

            image = ip.load(datadir_df + name).astype('single')
            if image.max()>3e3:
                err_x, err_y = (image.argmax()//length, image.argmax()%length)
                image[err_x-3:err_x+4, err_y-3:err_y+4] = 0


            image[image<0] = 0
            image = image**1.4

            if i_step == 0:
                image,xy = im.centering_com(image, iw_data['pp_df_int'])
            else:
                image,_ = im.centering_com(image, iw_data['pp_df_int'], xy)
                image,_ = im.centering_xcor(image, image0)
            image[image<0] = 0
            image = ip.normalize(image,(0,1.8))

            ds[i_px, i_step] = np.copy(image) 

            if i_step == 0:
                image0 = np.copy(image)

    arr = np.copy(ds[-1])
    ds[30:] = ds[29:-1]
    ds[29] = arr

    print('done')


