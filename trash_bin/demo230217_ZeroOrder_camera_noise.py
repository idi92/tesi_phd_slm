import numpy as np
from astropy.io import fits

def main():
    fdir = 'C:/Users/labot/Desktop/misure_tesi_slm/230217'
    fname_bg = '/230217bg_camera.fits'
    fname_RON = '/230217RON_camera.fits'
    
    def load_camera_background(fname):
        header = fits.getheader(fname)
        hduList = fits.open(fname)
        mean_bg = hduList[0].data
        sigma_bg = hduList[1].data
        
        Nframes = header['N_AV_FR']
        texp = header['T_EX_MS']
        return mean_bg, sigma_bg, Nframes, texp
    fname_lasercovered = fdir + fname_bg
    fname_camcovered = fdir + fname_RON
    bg1, sigma1,  Nframes1, texp1 =  load_camera_background(fname_lasercovered)
    bg2, sigma2,  Nframes2, texp2 =  load_camera_background(fname_camcovered)
    
    import matplotlib.pyplot as plt
    diff = bg1-bg2
    print('laser covered: (spatial) mean %g std %g'%( bg1.mean(), bg1.std()))
    print('camera covered: (spatial) mean %g std %g'% (bg2.mean(), bg2.std()))
    print('difference: (spatial) mean %g std %g'% (diff.mean(), diff.std()))
    
    plt.figure()
    plt.imshow(diff, cmap = 'jet', vmin=diff.min(), vmax=diff.max())
    plt.colorbar()
    plt.title('Flux difference')