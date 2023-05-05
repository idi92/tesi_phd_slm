import numpy as np 
import matplotlib.pyplot as plt
from astropy.io import fits
import pysilico
from tesi_slm.camera_masters import CameraMastersAnalyzer
from tesi_slm import my_tools
import time

def main(fname, fname_masters, Npoints = 100, t_sleep = 30,  Nframes = 100, texp = 0.3, fwhm_dl= 31.5, hs=100 ):
    
    texp_master, N_frames, master_dark, master_background = CameraMastersAnalyzer.load_camera_masters(fname_masters)
    if(texp != texp_master):
            print('WARNING: the selected exposure time (t_exp = %g ms) is different from the '\
                  'one used to measure dark and background (t_m = %g ms)\n'\
                  'NOTE: if t_m = 0s, image reduction is not performed.'
                  %(texp, texp_master))
    
    cam = pysilico.camera('localhost', 7100)
    cam.setExposureTime(texp)
    
    
    time_vector = np.zeros(Npoints)
    dead_time = np.zeros(Npoints)
    I = np.zeros(Npoints)
    err_I =  np.zeros(Npoints)
    fwhm_x = np.zeros(Npoints)
    err_fwhm_x =  np.zeros(Npoints)
    fwhm_y = np.zeros(Npoints)
    err_fwhm_y =  np.zeros(Npoints)
    amps = np.zeros(Npoints)
    err_amps =  np.zeros(Npoints)
    
    t0 = time.time()
    
    for idx in range(Npoints):
        print(idx)
        ima_cube = cam.getFutureFrames(Nframes)
        
        time_vector[idx] = time.time() - t0
        
        ima_cube = ima_cube.toNumpyArray()
        
        
        
        clean_ima = my_tools.clean_cube_images(ima_cube, master_dark, master_background)
        
        ymax, xmax = my_tools.get_index_from_image(clean_ima)
        
        cut_ima = my_tools.cut_image_around_coord(clean_ima, ymax, xmax, hs)
        
        par, err = my_tools.execute_gaussian_fit_on_image(cut_ima, fwhm_dl, fwhm_dl, False)
        I[idx] = cut_ima.sum()
        amps[idx] = par[0]
        fwhm_x[idx] = par[3]
        fwhm_y[idx] =  par[4]
        
        err_I[idx] = 0
        err_amps[idx] = err[0]
        err_fwhm_x[idx] = err[3]
        err_fwhm_y[idx] =  err[4]
        
        time.sleep(t_sleep)
        dead_time[idx] = time.time() + t0
        
    plt.figure()
    plt.plot(time_vector, I, 'r.-')
    plt.xlabel('time [s]')
    plt.ylabel('Counts in ROI [ADU]')
    #plt.legend(loc='best')
    plt.grid('--', alpha=0.3)
    
    plt.figure()
    plt.plot(time_vector, fwhm_x, 'b.--', label = 'FWHM-x')
    plt.plot(time_vector, fwhm_y, 'r.--', label = 'FWHM-y')
    plt.errorbar(time_vector, fwhm_x, err_fwhm_x, fmt='.b')
    plt.errorbar(time_vector, fwhm_y, err_fwhm_y, fmt='.r')
    plt.hlines(fwhm_dl, 0, time_vector.max(),'k', '--', label = 'Diffraction limit')
    plt.xlabel('time [s]')
    plt.ylabel('FWHM [pixel]')
    plt.legend(loc='best')
    plt.grid('--', alpha=0.3)
    
    plt.figure()
    plt.plot(time_vector, amps, 'm.--', label = 'texp=%g ms'%texp)
    plt.errorbar(time_vector, amps, err_amps, fmt='.m')
    plt.xlabel('time [s]')
    plt.ylabel('Peak Amplitude [ADU]')
    plt.legend(loc='best')
    plt.grid('--', alpha=0.3)
    
    fits.writeto(fname, I)
    fits.append(fname, fwhm_x)
    fits.append(fname, fwhm_y)
    fits.append(fname, amps)
    fits.append(fname, time_vector)
    fits.append(fname, dead_time)
    fits.append(fname, np.array([Npoints, t_sleep, Nframes, texp, fwhm_dl, hs]))
    fits.append(fname, err_I)
    fits.append(fname, err_fwhm_x)
    fits.append(fname, err_fwhm_y)
    fits.append(fname, err_amps)
    return time_vector, I, err_I, fwhm_x,err_fwhm_x, fwhm_y,err_fwhm_y, amps,err_amps
    
def load_and_plot(fname_laser_stab):
    data_file = fits.open(fname_laser_stab)
    I = data_file[0].data
    fwhm_x = data_file[1].data
    fwhm_y = data_file[2].data
    amps = data_file[3].data
    time_vector = data_file[4].data
    dead_time = data_file[5].data
    Npoints, t_sleep, Nframes, texp, fwhm_dl, hs = data_file[6].data
    
    try:
        err_I = data_file[7].data
        err_fwhm_x = data_file[8].data
        err_fwhm_y = data_file[9].data
        err_amps = data_file[10].data
    except:
        print('WORNING: err bars missing from data!')
        return I, fwhm_x, fwhm_y, amps, time_vector, dead_time, data_file[6].data
    
    return I, fwhm_x, fwhm_y, amps, time_vector, dead_time, data_file[6].data, \
        err_I, err_fwhm_x, err_fwhm_y, err_amps
    