from tesi_slm import psf_on_camera_optimizer
import numpy as np 
from astropy.io import fits

def main():
    cam, mirror = psf_on_camera_optimizer.create_devices()
    poco = psf_on_camera_optimizer.PsfOnCameraOptimizer(cam, mirror)
    
    fdir = 'C:/Users/labot/Desktop/misure_tesi_slm/230217'
    fname = '/230217bg_camera.fits'
    fname_bg = fdir + fname
    poco.load_camera_background(fname_bg)
    poco.change_circular_mask(centerYX=(576,960), RadiusInPixel=555)
    
    coeff = [0, 0, -230e-9, -80e-9, 140e-9, -30e-9, -20e-9, 0, 0, -30e-9]
    poco._write_zernike_on_slm(coeff)
    cam.setExposureTime(poco._texp_bg)
    Nframes = 1000
    Npsf = poco.get_frames_from_camera(Nframes)
    psf_mean, psf_err = poco.get_mean_and_std_from_frames(Npsf)
    psf_mean = poco._subtract_background_from_image(psf_mean)
    
    import matplotlib.pyplot as plt
    plt.subplots(1, 2, sharex=True, sharey=True)
    plt.subplot(1, 2, 1)
    plt.title('mean psf')
    plt.imshow(psf_mean, cmap = 'jet', vmax=psf_mean.max(), vmin = psf_mean.min())
    plt.colorbar(orientation="horizontal", pad = 0.05)
    plt.subplot(1, 2, 2)
    plt.title('std psf')
    plt.imshow(psf_err, cmap = 'jet', vmax=psf_err.max(), vmin = psf_err.min())
    plt.colorbar(orientation="horizontal", pad=0.05)
    
    fnamepsf = fdir + '/230217psf_bg_sub_0125ms_1000frames.fits'
    hdr = fits.Header()
    hdr['T_EX_MS'] = poco._texp_bg
    hdr['N_AV_FR'] = Nframes
    fits.writeto(fnamepsf, psf_mean, hdr)
    fits.append(fnamepsf, psf_err)

def load_psf(psf_fname):
    header = fits.getheader(psf_fname)
    hduList = fits.open(psf_fname)
    psf_mean = hduList[0].data
    psf_err = hduList[1].data
        
    Nframes = header['N_AV_FR']
    texp = header['T_EX_MS']
    return psf_mean, psf_err, Nframes, texp