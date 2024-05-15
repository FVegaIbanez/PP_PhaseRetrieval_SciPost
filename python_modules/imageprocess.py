import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TkAgg')
from matplotlib.widgets import Slider, Button
import hyperspy.api as hs

def loadSingleFile(filename, idx):
    file = hs.load(filename, lazy=True)
    if isinstance(file, list):
        file = file[1]
    if filename[-4:] == '.emi':
        if len(file.data.shape) == 2:
            return file.data.compute()
        else:
            return (file.data[idx]).compute()
    elif filename[-4:] == '.dm4':
        return file.data.compute()

# def loadSingleFile(filename):
#     return hs.load(filename).data[0]

def load(filename, idx=0):
    if isinstance(filename, list):
        return [ loadSingleFile(name, idx) for name in filename ]
    elif isinstance(filename, str):
        return loadSingleFile(filename, idx)

def shift(arr: np.ndarray, x, y):
    sub_x = x - round(x)
    sub_y = y - round(y)
    x = round(x)
    y = round(y)
    if x==0 and y==0:
        arr_shifted = subPixelShift(arr, (sub_x, sub_y))
    elif x==0:
        arr_shifted = np.zeros(arr.shape)
        if y > 0:
            arr_shifted[:,y:] = arr[:,:-1*y]
        elif y < 0:
            arr_shifted[:,:y] = arr[:,-1*y:]
        arr_shifted = subPixelShift(arr_shifted, (sub_x, sub_y))
    elif y==0:
        arr_shifted = np.zeros(arr.shape)
        if x > 0:
            arr_shifted[x:,:] = arr[:-1*x,:]
        elif x < 0:
            arr_shifted[:x,:] = arr[-1*x:,:]
        arr_shifted = subPixelShift(arr_shifted, (sub_x, sub_y))
    else:
        arr_shifted = np.zeros(arr.shape)
        if x > 0:
            arr_shifted[x:,:] = arr[:-1*x,:]
        elif x < 0:
            arr_shifted[:x,:] = arr[-1*x:,:]
        if y > 0:
            arr_shifted[:,y:] = arr_shifted[:,:-1*y]
            arr_shifted[:,:y] = 0
        elif y < 0:
            arr_shifted[:,:y] = arr_shifted[:,-1*y:]
            arr_shifted[:,y:] = 0
        arr_shifted = subPixelShift(arr_shifted, (sub_x, sub_y))
    return arr_shifted

def subPixelShift(arr, vec):
    if vec[0] != 0 and vec[1] !=0:
        lx, ly = arr.shape
        x_kernel, y_kernel = cartKernel((lx,ly))
        x_kernel -= lx//2
        y_kernel -= ly//2
        x_kernel = np.exp( -vec[0]*1j*2*np.pi/lx * x_kernel )
        y_kernel = np.exp( -vec[1]*1j*2*np.pi/ly * y_kernel )
        arr = ifft2d( fft2d(arr) * x_kernel * y_kernel )
    return np.real(arr)

def bin(arr: np.ndarray, bin_times: int):
    lx, ly = arr.shape[:1]
    n_bin_x = lx // bin_times
    n_bin_y = ly // bin_times
    arr = arr[0:-1*lx%bin_times, 0:-1*ly%bin_times]
    return np.reshape(arr, [n_bin_x, bin_times, n_bin_y, bin_times]).sum(3).sum(1)

def fft2d(arr, b_norm=False, b_safe=False):
    if b_safe:
        big_arr = np.zeros((arr.shape[0]*3, arr.shape[1]*3))
        arr=insertCenter(big_arr, arr)
    out = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(arr)))
    if b_norm:
        out = out / arr.shape[0]**0.5 * arr.shape[1]**0.5
    return out

def ifft2d(arr, b_norm=False, b_safe=False):
    out =  np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(arr)))
    if b_norm:
        out = out * arr.shape[0]**0.5 * arr.shape[1]**0.5
    if b_safe:
        out = out[arr.shape[0]//3:-arr.shape[0]//3, arr.shape[1]//3:-arr.shape[1]//3]
    return out

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def cartKernel(shape):
    x = np.arange(shape[1])
    y = np.arange(shape[0])
    x_kernel, y_kernel,  = np.meshgrid(x, y)
    return x_kernel, y_kernel

def polarKernel(shape):
    x_kernel, y_kernel = cartKernel(shape)
    rho_kernel, phi_kernel = cart2pol(x_kernel-shape[0]//2, y_kernel-shape[1]//2)
    return rho_kernel, phi_kernel

def com(arr: np.ndarray, reference='center', b_normalize=True):
    x_kernel, y_kernel = cartKernel(arr.shape)
    if b_normalize:
        dose = arr.sum()
    else:
        dose = 1
    comx = (arr*x_kernel / dose).sum()
    comy = (arr*y_kernel / dose).sum()
    if reference == 'center':
        return [comx-arr.shape[0]//2, comy-arr.shape[1]//2]
    elif reference == 'zero':
        return [comx, comy]

def normalize(arr, minmax=(0, 1), b_binary=False, b_unitary=False):
    minmax = ( float(min(minmax)), float(max(minmax)) )
    arr = np.copy(arr).astype('float')
    arr *= (minmax[1] - minmax[0]) / (arr.max() - arr.min())
    arr = arr - arr.min() + minmax[0]
    if b_binary:
        arr[arr >= arr.mean()] = minmax[1]
        arr[arr < arr.mean()] = minmax[0]
    if b_unitary:
        arr /= arr.sum()
    return arr

def insertCenter(arr_big, arr_small, b_reverse=False):
    bx, by = arr_big.shape
    sx, sy = arr_small.shape
    arr_big = arr_big.astype(arr_small.dtype)
    if b_reverse:
        return arr_big[bx//2-sx//2 : bx//2+(sx-1)//2+1, 
            by//2-sy//2 : by//2+(sy-1)//2+1]
    else:
        arr_big[bx//2-sx//2 : bx//2+(sx-1)//2+1, 
            by//2-sy//2 : by//2+(sy-1)//2+1] = arr_small
        return arr_big

def matchShape(arr, shape):
    sx, sy = shape
    lx, ly = arr.shape
    if lx > sx:
        arr = insertCenter(arr, np.zeros((sx, ly)), b_reverse=True)
    elif lx < sx:
        arr = insertCenter(np.zeros((sx, ly)), arr)
    if ly > sy:
        arr = insertCenter(arr, np.zeros((sx, sy)), b_reverse=True)
    elif ly < sy:
        arr = insertCenter(np.zeros((sx, sy)), arr)
    return arr




def resize(arr, ratio, b_binary=False):
    minmax = [arr.min(), arr.max()]
    minmax = np.real(minmax)
    if ratio < 1:
        dx = arr.shape[0]//2 - round(arr.shape[0]*ratio)//2
        dy = arr.shape[1]//2 - round(arr.shape[1]*ratio)//2
        arr_resized = np.real(ifft2d( fft2d(arr)[dx:dx+round(arr.shape[0]*ratio), dy:dy+round(arr.shape[1]*ratio)] ))
    elif ratio > 1:
        arr_resized = np.zeros( (round(arr.shape[0]*ratio), round(arr.shape[1]*ratio)) )
        arr_resized = np.real(ifft2d( insertCenter(arr_resized, fft2d(arr)) ))
    else:
        arr_resized = arr
    arr_resized = normalize(arr_resized, minmax=minmax)  
    if b_binary:
        arr_resized = normalize(arr_resized)
        arr_resized[arr_resized > 0.5] = 1
        arr_resized[arr_resized < 0.5] = 0
        arr_resized = normalize(arr_resized, (arr.min(), arr.max()))
    else:        
        arr_resized = normalize(arr_resized, (arr.min(), arr.max()))
    
    return np.real(arr_resized)

def houghCircle(arr, args, b_confrimation=False):
    img = normalize(arr, (0,255)).astype('uint8')

    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT_ALT, 1.5,
        minDist=arr.shape[0], param1=args[0], param2=args[1],
        minRadius=int(args[2]), maxRadius=int(args[3]))
    id_max = np.argmax(circles[0][:,2])
    if b_confrimation:
        img_u = cv2.UMat(img).get()
        for c in circles[0,:]:
            cv2.circle(img_u, (int(c[0]), int(c[1])), int(c[2]), (255, 255, 255), arr.shape[0]//100+5)
            cv2.circle(img_u, (int(c[0]), int(c[1])), 1, (255, 255, 255), arr.shape[0]//100+5)
        img_u = cv2.resize(img_u, (512, 512))     
        cv2.imshow('found circles', img_u)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return circles[0][id_max,:2]

def centering(arr, method, *args):
    if callable(method):        
        cx, cy = method(arr, args)
        return shift(arr, -cx, -cy)
    else:
        return shift(arr, method[0], method[1])

def makeMovie(img_set, fname):
    _, lx, ly = img_set.shape
    video = cv2.VideoWriter(fname, 0, 20, (lx, ly))
    for image in img_set:
        image = normalize(image, (0,255)).astype('uint8')
        img_u = cv2.UMat(image).get()
        video.write(img_u)
    video.release()
    cv2.destroyAllWindows()

def xcor(img1, img2):
    return np.real( ifft2d( np.conjugate(fft2d(img1)) * fft2d(img2) ) )

def localMaxMap(arr):
    xdiff1 = np.diff(arr, axis=0, prepend=np.expand_dims(arr[0,:],0))
    xdiff2 = np.diff(arr, axis=0, append=np.expand_dims(arr[-1,:],0))
    ydiff1 = np.diff(arr, axis=1, prepend=np.expand_dims(arr[:,0],1))
    ydiff2 = np.diff(arr, axis=1, append=np.expand_dims(arr[:,-1],1))
    return ((xdiff1>0)  * (xdiff2<0)) + ((ydiff1>0) * (ydiff2<0))
    
def show(img, b_block=False):
    if isinstance(img, list):
        n_img = len(img)
        _, ax = plt.subplots(1,n_img)
        for h in range(n_img):
            ax[h].imshow(img[h])
        plt.show(block=b_block)
    else:
        plt.figure(); plt.imshow(img); plt.show(block=b_block)

def rmBackground(img, ratio):
    val = img.min() + (img.max()-img.min())*ratio
    img[img<val] = val
    return img

def slider(img_list, b_block=False):
    fig, ax = plt.subplots()
    im = ax.imshow(img_list[0])
    fig.subplots_adjust(bottom=0.25)
    list_idx = np.arange(len(img_list))
    ax_idx = fig.add_axes([0.2, 0.1, 0.6, 0.1])
    slider = Slider(ax=ax_idx, valmin=0, valmax=len(img_list)-1, valinit=0, label=None, valstep=list_idx, closedmax=True)    
    def update(val):
        im.set_data(img_list[val])
        fig.canvas.draw_idle()
    slider.on_changed(update)
    plt.show()

def plot(img, b_block=False, b_same_fig=False):
    if isinstance(img, list):
        if b_same_fig:
            plt.figure() 
            for i in img:
                plt.plot(i)
            plt.show(block=b_block)
        else:
            n_img = len(img)
            _, ax = plt.subplots(1,n_img)
            for h in range(n_img):
                ax[h].plot(img[h])
            plt.show(block=b_block)
    else:
        plt.figure(); plt.plot(img); plt.show(block=b_block)

def scatter(img, b_block=False):
    if isinstance(img, list):
        n_img = len(img)
        _, ax = plt.subplots(1,n_img)
        for h in range(n_img):
            ax[h].scatter(img[h][0],img[h][1])
        plt.show(block=b_block)
    else:
        plt.figure(); plt.scatter(img[0],img[1]); plt.show(block=b_block)

def frqPatch(shape, frq_cutoff, amp_cutoff, mode='lowpass', seed=None):
    if seed is not None:
        np.random.seed(seed)
    img = fft2d(np.random.rand(shape[0],shape[1]))
    rho, _ = polarKernel(shape)
    cutoff = frq_cutoff * shape[0] / 2 
    if mode == 'lowpass':
        rho[rho>cutoff] = 0
        filter = rho
    elif mode == 'highpass':
        rho[rho<cutoff] = 0
        filter = rho
    elif mode == 'gauss-lowpass':
        gauss = normalize(1/(2*np.pi)**2/cutoff * np.exp(-rho**2/2/cutoff**2))
        filter = gauss
    elif mode == 'gauss-highpass':
        gauss = normalize(1/(2*np.pi)**2/cutoff * np.exp(-rho**2/2/cutoff**2))
        filter = (1 - gauss)
    img = normalize( np.real( ifft2d(img * filter) ) )
    img[img<amp_cutoff] = 0
    return img
