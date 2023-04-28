import numpy as np 
import matplotlib.pyplot as plt
from astropy.io import fits
import pysilico
from tesi_slm.camera_masters import CameraMastersAnalyzer
from tesi_slm import my_tools
import time


def main(fname, fname_masters, Nframes, texp, fwhm_dl=3.26, hs=25):
    
    texp_master, N_frames, master_dark, master_background = CameraMastersAnalyzer.load_camera_masters(fname_masters)
    if(texp != texp_master):
            print('WARNING: the selected exposure time (t_exp = %g ms) is different from the '\
                  'one used to measure dark and background (t_m = %g ms)\n'\
                  'NOTE: if t_m = 0s, image reduction is not performed.'
                  %(texp, texp_master))
    
    cam = pysilico.camera('localhost', 7100)
    cam.setExposureTime(texp)
    start_time = time.time()
    ima = cam.getFutureFrames(Nframes)
    end_time = time.time()
    ima = ima.toNumpyArray()
    dt = end_time - start_time
    print('dt = %g s'%dt)
    clean_cube = np.zeros(ima.shape)
    for n in range(Nframes):
        image = ima[:, :, n]
        clean_cube[:, :, n] = my_tools.clean_image(image, master_dark, master_background)
    
    clean_ima = clean_cube.mean(axis = 2)
    
    ymax, xmax = my_tools.get_index_from_image(clean_ima)
    
    I = np.zeros(Nframes)
    fwhm_x = np.zeros(Nframes)
    fwhm_y = np.zeros(Nframes)
    amps = np.zeros(Nframes)
    
    for n in range(Nframes):
        cut_ima = my_tools.cut_image_around_coord(clean_cube[:,:,n], ymax, xmax, hs)
        par, err = my_tools.execute_gaussian_fit_on_image(cut_ima, fwhm_dl, fwhm_dl, False)
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
    plt.hlines(fwhm_dl,0, Nframes,'k', '--', label = 'Diffraction limit')
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
    fits.append(fname, np.array([dt,texp,Nframes]))
    
    return I, fwhm_x, fwhm_y, amps

def load_and_plot(fname_laser_stab):
    data_file = fits.open(fname_laser_stab)
    I = data_file[0].data
    fwhm_x = data_file[1].data
    fwhm_y = data_file[2].data
    amps = data_file[3].data
    dt, texp, Nframes = data_file[4].data
    
    plt.figure()
    plt.plot(I, 'r.--', label = 'texp=%g ms'%texp)
    plt.xlabel('Frames')
    plt.ylabel('Counts in ROI [ADU]')
    plt.legend(loc='best')
    plt.grid('--',alpha=0.3)
    
    plt.figure()
    plt.plot(fwhm_x, 'b.--', label = 'FWHM-x')
    plt.plot(fwhm_y, 'r.--', label = 'FWHM-y')
    #plt.hlines(fwhm_dl,0, Nframes,'k', '--', label = 'Diffraction limit')
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
    
    I_mean = I.mean()
    I_err = I.std()
    fwhm_x_mean = fwhm_x.mean()
    fwhm_x_err = fwhm_x.std()
    fwhm_y_mean = fwhm_y.mean()
    fwhm_y_err = fwhm_y.std()
    amps_mean = amps.mean()
    amps_err = amps.std()
    print('dt = %g s'%dt)
    print('texp = %g ms' %texp)
    print('Nframes = %d'%Nframes)
    print('I = {} +/- {}  ADU'.format(I_mean,I_err))
    print('FWHM-X = {} +/- {}  pixel'.format(fwhm_x_mean,fwhm_x_err))
    print('FWHM-Y = {} +/- {}  pixel'.format(fwhm_y_mean,fwhm_y_err))
    print('amp = {} +/- {}  ADU'.format(amps_mean,amps_err))
    
    return I, fwhm_x, fwhm_y, amps, dt, texp, Nframes
    