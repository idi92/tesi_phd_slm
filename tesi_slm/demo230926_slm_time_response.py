
from arte.types.mask import CircularMask
from arte.utils.zernike_generator import ZernikeGenerator
from tesi_slm import my_tools
import numpy as np
from plico_dm import deformableMirror
import time
def get_cmd_vectors(c2_m_rms = 30e-6):
    
    frameshape = (1152, 1920)
    radius = 571
    centeryx = (571, 875)
    cmask_obj = CircularMask(frameshape, radius, centeryx)
    #building a zernike over the pupil
    zg = ZernikeGenerator(cmask_obj)
    #tilt1 = my_tools.reshape_map2vector(c2*z2)
    #tilt2 = my_tools.reshape_map2vector(-c2*z2)
    fspoc = "C:\\Users\\labot\\Desktop\\misure_tesi_slm\\non_common_path_abs\\230911\\230919spoc_coef_matrix_flat_texp1.0ms_10times_v4.fits"
    hh,dd = my_tools.open_fits_file(fspoc)
    coeff= dd[0].data.mean(axis=1)
    coeff1 = coeff.copy()
    coeff1[0] = c2_m_rms
    coeff2 = coeff.copy()
    coeff2[0] = - c2_m_rms
    
    image_to_display1 = np.zeros(frameshape)
    image_to_display2 = np.zeros(frameshape)
    image_to_display1 = np.ma.array(data = image_to_display1, mask = cmask_obj.mask(), fill_value = 0)
    image_to_display2 = np.ma.array(data = image_to_display2, mask = cmask_obj.mask(), fill_value = 0)
    for j, aj in enumerate(coeff1):
        Zj = zg.getZernike(j + 2)
        image_to_display1 += aj * Zj
    for j, aj in enumerate(coeff2):
        Zj = zg.getZernike(j + 2)
        image_to_display2 += aj * Zj
    
    cmd_vector1 = my_tools.reshape_map2vector(image_to_display1)
    cmd_vector2 = my_tools.reshape_map2vector(image_to_display2)
    
    return cmd_vector1, cmd_vector2

def execute(command1, command2, tsleep_ms):
    
    idx = 0
    mirror = deformableMirror('localhost', 7000)
    t = tsleep_ms*1e-3
    while(idx <=500):
        time.sleep(t)
        mirror.set_shape(command1)
        time.sleep(t)
        mirror.set_shape(command2)
        idx+=1
        
def save_cube_ima(cube_ima, fname, status, texp, frame_rate, Nframes):
    from astropy.io import fits
    hdr = fits.Header()
    hdr['T_EX_MS'] = texp
    hdr['N_AV_FR'] = Nframes
    hdr['FR_R'] = frame_rate
    hdr['ST_DATA'] = status  
    fits.writeto(fname , cube_ima, hdr)

def demo_lorenzo():
    fname = "C:\\Users\\labot\\Desktop\\misure_tesi_slm\\slm_time_response\\230929\\230929cubeima_flat_c2_20urms.fits"
    fname_masters = "C:\\Users\\labot\\Desktop\\misure_tesi_slm\\camera_masters\\230908cma_masters_texp1.0ms.fits"
    mh, md = my_tools.open_fits_file(fname_masters)
    master_bias = md[0].data
    master_background = md[1].data
    hh,dd = my_tools.open_fits_file(fname)
    cube_ima_raw = dd[0].data
    clean_ima = my_tools.get_clean_cube_images(cube_ima_raw, master_bias, master_background)
    
    import matplotlib.pyplot as plt
    plt.figure(44)
    plt.plot(clean_ima[:,:,41].sum(axis=0))
    plt.plot(clean_ima[:,:,39].sum(axis=0))
    plt.plot(clean_ima[:,:,40].sum(axis=0))
    
    clean_ima[460:510,650:1150].sum(axis=(0,1))
    
    clean_ima[460:510,650:1150,39:42].sum(axis=(0,1))
    #array([36487., 32241., 31017.])

    clean_ima[460:510,750:1050,39:42].sum(axis=(0,1))
    #array([-4754.5, -2495.5, -5886.5])
    clean_ima[470:500,680:710,39:42].sum(axis=(0,1))
    #array([40956.,  8138.,  1152.])
    clean_ima[470:500,1080:1110,39:42].sum(axis=(0,1))
    #array([ -260.5, 25550.5, 37119.5])

    clean_ima[470:500,680:710,39:42].sum(axis=(0,1))-1410
    #array([39546.,  6728.,  -258.])

    clean_ima[470:500,1080:1110,39:42].sum(axis=(0,1))
    #array([ -260.5, 25550.5, 37119.5])

    a=clean_ima[470:500,680:710,39:42].sum(axis=(0,1))-1410
    b=clean_ima[470:500,1080:1110,39:42].sum(axis=(0,1))
    #a+b
    #array([39285.5, 32278.5, 36861.5])
    
def show_intenity_profiles_vs_time():
    
    fname = "C:\\Users\\labot\\Desktop\\misure_tesi_slm\\slm_time_response\\230929\\230929cubeima_flat_c2_20urms.fits"
    fname_masters = "C:\\Users\\labot\\Desktop\\misure_tesi_slm\\camera_masters\\230908cma_masters_texp1.0ms.fits"
    mh, md = my_tools.open_fits_file(fname_masters)
    master_bias = md[0].data
    master_background = md[1].data
    hh,dd = my_tools.open_fits_file(fname)
    cube_ima_raw = dd[0].data
    clean_ima = my_tools.get_clean_cube_images(cube_ima_raw, master_bias, master_background)
    
    sumI0 = clean_ima[:,:,39].sum(axis=0)
    max_sumI0 = sumI0.max() 
    sumI0 = sumI0/max_sumI0
    sumI1 = clean_ima[:,:,40].sum(axis=0)/max_sumI0
    sumI2 = clean_ima[:,:,41].sum(axis=0)/max_sumI0
    
    import matplotlib.pyplot as plt
    plt.figure(44)
    plt.plot(clean_ima[:,:,41].sum(axis=0), label = '$t_2$')
    plt.plot(clean_ima[:,:,39].sum(axis=0), label = '$t_0$')
    plt.plot(clean_ima[:,:,40].sum(axis=0), label = '$t_1$')
    plt.ylabel('Sum over frame columns Intesity profile [ADU]')
    plt.xlabel('pixel')
    plt.legend(loc='best')
    plt.grid('--', alpha = 0.3)
    
    plt.subplots(3, 1, sharex=True, sharey = True)
    plt.subplot(3,1,1)
    plt.plot(sumI0, label='$t_0$')
    #plt.xlabel('pixel')
    #plt.ylabel('Normalized Intesity')
    plt.legend(loc = 'best')
    plt.grid('--', alpha = 0.3)
    plt.subplot(3,1,2)
    plt.plot(sumI1, label='$t_1$')
    #plt.xlabel('pixel')
    plt.ylabel('Normalized Intesity Profile')
    plt.legend(loc = 'best')
    plt.grid('--', alpha = 0.3)
    plt.subplot(3,1,3)
    plt.plot(sumI2, label='$t_2$')
    plt.xlabel('pixel')
    #plt.ylabel('Normalized Intesity')
    plt.legend(loc = 'best')
    plt.grid('--', alpha = 0.3)
 