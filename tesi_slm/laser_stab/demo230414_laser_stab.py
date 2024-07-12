import numpy as np 
import matplotlib.pyplot as plt
from astropy.io import fits
from tesi_slm import sharp_psf_on_camera
from tesi_slm import my_tools


def main(fname, fname_masters, Nframes, texp):
    cam, mirror  = sharp_psf_on_camera.create_devices()
    best_coeff_std = np.array([ 0., 0., -4.e-08, -8.e-08, -8.e-08, -4.e-08,  4.e-08, 4.e-08,  0.,  0.])
    #fname_masters = 'C:\\Users\\labot\\Desktop\\misure_tesi_slm\\230414\\230414cma_masters_0p035ms.fits'
    spoc = sharp_psf_on_camera.SharpPsfOnCamera(cam,mirror,fname_masters)
    spoc.change_circular_mask(centerYX=(550,853),RadiusInPixel=569)
    spoc.write_zernike_on_slm(best_coeff_std)
    cam.setExposureTime(texp)
    ima = cam.getFutureFrames(Nframes).toNumpyArray()
    
    
    clean_cube = np.zeros(ima.shape)
    for n in range(Nframes):
        image = ima[:, :, n]
        clean_cube[:, :, n] = my_tools.clean_image(image, spoc._master_dark, spoc._master_background)
    
    clean_ima = clean_cube.mean(axis = 2)
    
    ymax, xmax = my_tools.get_index_from_image(clean_ima)
    
    I = np.zeros(Nframes)
    fwhm_x = np.zeros(Nframes)
    fwhm_y = np.zeros(Nframes)
    amps = np.zeros(Nframes)
    
    for n in range(Nframes):
        cut_ima = my_tools.cut_image_around_coord(clean_cube[:,:,n], ymax, xmax, 25)
        par, err = my_tools.execute_gaussian_fit_on_image(cut_ima, 3.26, 3.26, False)
        I[n] = cut_ima.sum()
        amps[n] = par[0]
        fwhm_x[n] = par[3]
        fwhm_y[n] =  par[4]
        
    plt.figure()
    plt.plot(I, 'r.--', label = 'texp=%g ms'%texp)
    plt.xlabel('Frames')
    plt.ylabel('Counts in ROI [ADU]')
    plt.legend(loc='best')
    plt.grid('--',alpha=0.3)
    
    plt.figure()
    plt.plot(fwhm_x, 'b.--', label = 'FWHM-x')
    plt.plot(fwhm_y, 'r.--', label = 'FWHM-y')
    plt.hlines(3.26,0, Nframes,'k', '--', label = 'Diffraction limit')
    plt.xlabel('Frames')
    plt.ylabel('FWHM [pixel]')
    plt.legend(loc='best')
    plt.grid('--',alpha=0.3)
    
    plt.figure()
    plt.plot(amps, 'm.--', label = 'texp=%g ms'%texp)
    plt.xlabel('Frames')
    plt.ylabel('Peak Amplitude [ADU]')
    plt.legend(loc='best')
    plt.grid('--',alpha=0.3)
    
    
    fits.writeto(fname, I)
    fits.append(fname, fwhm_x)
    fits.append(fname, fwhm_y)
    fits.append(fname, amps)
    
    return I, fwhm_x, fwhm_y, amps