import cv2
from scipy import ndimage
import numpy as np
import tools.imageprocess as ip
import tools.physics as ph


class PhasePlate:
    def __init__(self, arg):
        self.raw_img        = self.loadImage(arg)
        self.b_wave_built   = False
        self.wave           = self.raw_img**0.5
        self.length         = self.raw_img.shape[0]
        self.fill_ratio     = 1
        self.rot_angle      = 0
        self.b_property     = False
        self._phase         = np.ones(self.raw_img.shape, dtype='complex64')
        self.update()
        self.idx:           list
        self.idx_map:       np.ndarray
        self.idx_list:      list
        self.conv_angle:    float
        self.mrad_per_px:   float
        self.rAng_per_px:   float
        self.Ang_per_px:    float
        self.max_angle:     float
        self.obj_size:      float
        self.mode:          str

    def loadImage(self, arg):
        try:
            if isinstance(arg, str):
                img = cv2.imread(arg, cv2.IMREAD_GRAYSCALE).astype('float')
            elif isinstance(arg, np.ndarray):
                img = arg.astype('float')
            self.length = img.shape[0]
            self.fill_ratio = 1.0
            return ip.normalize(img, b_unitary=True)
        except:
            print('\nImage load fail, please provide path or numpy array.\n')

    def setFrame(self, length, fill_ratio, rot_angle):
        self.length     = int(length)
        self.fill_ratio = fill_ratio
        self.rot_angle  = rot_angle
        self.update()

    # def magnifyCurrent(self, magnification):
    #     self.fill_ratio = np.min( [1, self.fill_ratio*magnification] )
    #     if self.b_property:
    #         self.reassign()
    #     self.update()

    def rotateCurrent(self, rot_angle):
        self.rot_angle += rot_angle
        self.update()
     
    def update(self):
        pp_length = round(self.length * self.fill_ratio)
        self.wave = ndimage.zoom(self.raw_img, pp_length/self.raw_img.shape[0])
        self.wave[self.wave<0]=0
        self.wave = ip.normalize(self.wave**2, b_binary= True, b_unitary=True)
        self.wave = ip.insertCenter(np.zeros((self.length, self.length)), self.wave)
        self.wave = ndimage.rotate(self.wave, self.rot_angle*180/np.pi, reshape=False)
        self.wave[self.wave<0]=0
        self.wave = ip.normalize(self.wave**2, b_binary= True, b_unitary=True)
        # self.wave = ndimage.binary_opening(self.wave, structure=np.ones((pp_length//200,pp_length//200))).astype(int)
        # self.wave = ndimage.binary_closing(self.wave, structure=np.ones((pp_length//200,pp_length//200))).astype(int)
        # self.wave = ip.normalize(self.wave, b_binary= True, b_unitary=True)
        self.indexPixels()

    @ property
    def phase(self):
        return np.angle(self._phase)

    @ phase.setter
    def phase(self, phase, m=None, b_average=False):
        '''
        Set the phase of the wave

            Parameter
            ---------
                phase: list or np.ndarray with datatype float
                    Define value of each pixel with a list, or define the phase for the whole wave with ndarray
                m:  np.ndarray
                    Define specific grouping method that can be represented as a matrix
                b_average: bool
                    Take the average phase inside each pixel. Default: False
        '''
        self._phase = np.ones(self.wave.shape, dtype='complex64')
        if isinstance(phase, list):
            phase = np.array(phase)
        
        # input: phase image
        if phase.shape == self.wave.shape:
            phase = np.exp(1j*phase)
            if b_average:
                # for i in range(1, self.idx_map.max()+1):
                #     self._phase[self.idx_map==i] = \
                #         np.exp( 1j * np.angle(phase[self.idx_map==i].sum()) )
                for map in self.map_list[1:]:
                    self._phase[map] = np.exp(1j*np.angle(phase[map].sum()))
            else:
                self._phase = phase
        # input: list of phase value
        else:
            if m is None:
                phase_list = phase 
            else:
                phase_list = phase @ m
            # for i in range(1, self.idx_map.max()+1):
            #     self._phase[self.idx_map==i] = np.exp(1j*phase_list[i-1])
            for i, map in enumerate(self.map_list[1:]):
                self._phase[map] = np.exp(1j*phase_list[i])
        self.wave = np.abs(self.wave) * self._phase
            
    def assignProperty(self, arg, E0, mode='angle'):
        self.b_property = True
        self.E0 = E0
        self.mode = mode
        if mode == 'angle':      
            self.conv_angle     = arg 
            self.mrad_per_px    = arg * 2 / round(self.length * self.fill_ratio)
            self.rAng_per_px    = ph.mrad_2_rAng(E0, self.mrad_per_px)
            self.max_angle      = self.mrad_per_px * self.length / 2 
            self.space_size     = self.rAng_per_px ** -1
            self.Ang_per_px     = self.space_size / self.length
        elif mode == 'size':
            self.space_size     = arg
            self.Ang_per_px     = arg / self.length / self.fill_ratio
            self.rAng_per_px    = 1 / arg
            self.mrad_per_px    = ph.rAng_2_mrad(E0, self.rAng_per_px)
            self.max_angle      = self.mrad_per_px * self.length / 2
            self.conv_angle     = self.mrad_per_px * round(self.length * self.fill_ratio) / 2

    def reassign(self):
        if self.mode == 'angle':
            self.mrad_per_px    = self.conv_angle * 2 / round(self.length * self.fill_ratio)
            self.rAng_per_px    = ph.mrad_2_rAng(self.E0, self.mrad_per_px)
            self.max_angle      = self.mrad_per_px * self.length / 2 
            self.space_size     = self.rAng_per_px ** -1
            self.Ang_per_px     = self.space_size / self.length
        elif self.mode == 'size':
            self.conv_angle     = self.mrad_per_px * round(self.length * self.fill_ratio) / 2

    def defocus(self, dz, mode='samespace'):
        return ph.defocus(self.wave, dz, self.rAng_per_px, self.E0, mode=mode)

    def indexPixels(self):
        img = ip.normalize(np.abs(self.wave)**2, (0,255)).astype('uint8')
        ret, labels = cv2.connectedComponents(img)
        self.idx = [ np.argwhere(labels==i) for i in range(1,ret) ]
        self.idx_map = labels
        self.map_list = []
        for i in range(self.idx_map.max()+1):
            self.map_list.append(self.idx_map==i)

    def groupCart(self, x_bin_method='fd', y_bin_method='fd'):
        self.indexPixels()
        px_idx = np.arange(1, self.idx_map.max()+1)
        y, x = ip.cartKernel(self.wave.shape)
        x_score = [((self.idx_map==i) * x).mean() for i in range(1,self.idx_map.max()+1)]
        y_score = [((self.idx_map==i) * y).mean() for i in range(1,self.idx_map.max()+1)]
        x_group = self.grouping(px_idx, x_score, x_bin_method)
        y_group = self.grouping(px_idx, y_score, y_bin_method)
        return [x_group, self.index_translate_matrix(x_group)],\
            [y_group, self.index_translate_matrix(y_group)]

    def groupPol(self, rho_bin_method='fd', phi_bin_method='fd'):
        self.indexPixels()
        px_idx = np.arange(1, self.idx_map.max()+1)
        rho, phi = ip.polarKernel(self.wave.shape)
        rho_score = [((self.idx_map==i) * rho).mean() for i in range(1,self.idx_map.max()+1)]
        phi_score = [np.angle(((self.idx_map==i) * np.exp(1j*phi)).sum()) for i in range(1,self.idx_map.max()+1)]
        rho_group = self.grouping(px_idx, rho_score, rho_bin_method)
        phi_group = self.grouping(px_idx, phi_score, phi_bin_method)
        return [rho_group, self.index_translate_matrix(rho_group)],\
            [phi_group, self.index_translate_matrix(phi_group)]

    def groupGradient(self, direction, symmetry):
        '''
        Build a matrix to translate vector of grouped pixels to vector of individual pixels

        Parameter
        ---------
            direction: str
                The direction a twofold symmetry can be found, accept 'x' or 'y'
            symmetry: int
                An integer number of symmetry found on the phase plate.

        Return
        ------
            m: np.ndarray
                Return a matrix to translate shortened vector to normal vector
        '''
        x, y = ip.cartKernel(self.wave.shape)
        L2 = self.wave.shape[0]//2
        if direction == 'x':
            projector = [np.cos, np.sin]
        elif direction == 'y':
            projector = [np.cos, -1*np.sin]
        if symmetry//2 == 0:
            symmetry = symmetry//2

        d_list = []
        for map in self.map_list[1:]:
            distance = [(map*x).sum()/map.sum()-L2, (map*y).sum()/map.sum()-L2]
            d_list.append(distance)
        d_list = np.array(d_list)

        m = np.zeros((symmetry,self.idx_map.max()))
        for s in range(symmetry):
            theta = self.rot_angle + 2*np.pi*s/symmetry
            m[s,:] = abs(projector[0](theta)*d_list[:,0] + \
                projector[1](theta)*d_list[:,1])
        m /= abs(m).max(0)
        return np.array(m)


    @staticmethod
    def grouping(data, score, bin_method):
        _, bin = np.histogram(score, bins=bin_method)
        bin[-1] += 1
        idx = np.digitize(score, bins=bin)-1
        group = [[] for _ in range((len(bin)-1))]
        for i, id in enumerate(idx):
            group[id].append(data[i]-1)
        return group

    @staticmethod
    def index_translate_matrix(idx_group):
        '''
        Build a matrix to translate vector of grouped pixels to vector of individual pixels

        Parameter
        ---------
            idx_group: list
                List of groups of pixels

        Return
        ------
            m: np.ndarray
                A matrix for performing translation. Ex. vec_long = vec_short @ m
        '''
        n_ele = sum([len(group) for group in idx_group])
        m = np.zeros((len(idx_group), n_ele))
        for r in range(m.shape[0]):
            m[r, idx_group[r]] = 1
        return m