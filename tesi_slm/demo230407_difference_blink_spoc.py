from astropy.io import fits
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from arte.utils import zernike_generator

from arte.types.mask import CircularMask


 
def main(fname_blink, fname_spoc):
    im = Image.open(fname_blink)
    z6_blink = np.array(im, dtype = np.uint8)
    pippo = fits.open(fname_spoc)
    z6_spoc = pippo[0].data
    
    cmask_obj = CircularMask(frameShape=(1152,1920),maskCenter=(550,853),maskRadius=569)
    zg = zernike_generator.ZernikeGenerator(cmask_obj)
    z6 = zg.getZernike(6) * 2.e-6
    
    
    plt.subplots(2,1,sharex=True, sharey=True)
    plt.subplot(2,1,1)
    plt.imshow(z6_blink, cmap='gray')
    plt.colorbar()
    plt.subplot(2,1,2)
    plt.imshow(z6_spoc, cmap='gray')
    plt.colorbar()
    
    plt.figure()
    plt.plot(z6_blink[550,:], 'b-',label='Blink')
    plt.plot(z6_spoc[550,:], 'r-',label='PLICO')
    plt.plot((z6[550,:]/635e-9*256)%256, 'm--',label='unrounded')
    plt.legend(loc='best')
    plt.vlines(853,0, 256,'k','--')
    plt.xlabel('pixel along x-axis')
    plt.ylabel('gray value')
    plt.grid('--', alpha=0.3)
    
    plt.figure()
    plt.plot(z6_blink[:, 853], 'b-',label='Blink')
    plt.plot(z6_spoc[:, 853], 'r-',label='PLICO')
    plt.plot((z6[:,853]/635e-9*256)%256, 'm--',label='unrounded')
    plt.legend(loc='best')
    plt.vlines(550, 0, 256,'k','--')
    plt.xlabel('pixel along y -axis')
    plt.ylabel('gray value')
    plt.grid('--', alpha=0.3)
    

    cmask_obj = CircularMask(frameShape=(1152,1920),maskCenter=(550,853),maskRadius=569)
    zg = zernike_generator.ZernikeGenerator(cmask_obj)
    z6 = zg.getZernike(6) * 2.e-6
    
    
    plt.figure()
    plt.plot(z6_blink[550,:], 'b.-',label='Blink')
    plt.plot(z6_spoc[550,:], 'r.-',label='PLICO')
    plt.plot((z6[550,:]/635e-9*256)%256, 'm.--',label='unrounded')
    plt.legend(loc='best')
    plt.vlines(853,0, 256,'k','--')
    plt.xlabel('pixel along x-axis')
    plt.ylabel('gray value')
    plt.grid('--', alpha=0.3)
    
    plt.figure()
    plt.plot(z6_blink[:, 853], 'b.-',label='Blink')
    plt.plot(z6_spoc[:, 853], 'r.-',label='PLICO')
    plt.plot((z6[:,853]/635e-9*256)%256, 'm.--',label='unrounded')
    plt.legend(loc='best')
    plt.vlines(550, 0, 256,'k','--')
    plt.xlabel('pixel along y -axis')
    plt.ylabel('gray value')
    plt.grid('--', alpha=0.3)
    
    z_un = (z6/635e-9*256)%256
    print('value around center along x axis:')
    for idx in range(834, 866):
        print('index {}\t unrounded: {}\t blink: {}\t PLICO: {}'.format(
            idx-853, z_un[550,idx], z6_blink[550, idx], z6_spoc[550, idx]))
        
    print('value around center along y axis:')
    for idx in range(539, 561):
        print('index {}\t unrounded: {}\t blink: {}\t PLICO: {}'.format(
            idx-550, z_un[idx, 853], z6_blink[idx, 853], z6_spoc[idx, 853]))