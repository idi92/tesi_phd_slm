import pysilico
import numpy as np
from astropy.io import fits
import os

def main():
    cam = pysilico.camera('localhost', 7100)
    texp = 0.125
    cam.setExposureTime(exposureTimeInMilliSeconds = texp)
    Nframes = 1000
    bg_ima = cam.getFutureFrames(Nframes)
    bg = bg_ima.toNumpyArray()
    bg_mean = bg.mean(axis=2)
    bg_sigma = bg.std(axis=2)
    fdir = 'C:/Users/labot/Desktop/misure_tesi_slm/230216'
    fname = '/230216backgroung_camera.fits'
    #fpath = os.path.join(fdir, fname)
    fpath = fdir + fname
    hdr = fits.Header()
    hdr['T_EX_MS'] = texp
    hdr['N_AV_FR'] = Nframes
    fits.writeto(fpath, bg_mean, hdr)
    fits.append(fpath, bg_sigma)
    
    import matplotlib.pyplot as plt
    plt.figure()
    plt.subplots(1, 2, sharex=True, sharey=True)
    plt.subplot(1, 2, 1)
    plt.title('mean background')
    plt.imshow(bg_mean, cmap = 'jet', vmax=bg_mean.max(), vmin = bg_mean.min())
    plt.colorbar(orientation="horizontal", pad = 0.05)
    plt.subplot(1, 2, 2)
    plt.title('std background')
    plt.imshow(bg_sigma, cmap = 'jet', vmax=bg_sigma.max(), vmin = bg_sigma.min())
    plt.colorbar(orientation="horizontal", pad=0.05)