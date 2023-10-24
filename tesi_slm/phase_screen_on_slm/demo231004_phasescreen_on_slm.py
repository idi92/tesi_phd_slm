import numpy as np 
from arte.atmo.phase_screen_generator import PhaseScreenGenerator
from arte.utils.rebin import rebin
from tesi_slm import sharp_psf_on_camera
from time import sleep

def lorenzo_example():
    
    cam, mirror = sharp_psf_on_camera.create_devices()
    spoc = sharp_psf_on_camera.SharpPsfOnCamera(cam, mirror)
    spoc.change_circular_mask((571,875), 571)
    spoc.write_zernike_on_slm([15e-6])
    tilt = mirror.get_shape()
    
    Dpe = 8 # meters
    frame_size_in_px = (1152,1920)
    R_in_px = 571 # radius of slm circular mask
    screen_size_in_m = 0.5*Dpe/R_in_px * frame_size_in_px[1]
    outer_scale = 100 # meters
    seed = 98752546
    #crea un cubo di phase screens quadrati
    psg = PhaseScreenGenerator(frame_size_in_px[1], screen_size_in_m, outer_scale, seed)
    Nps = 10 # number of phase screens short exposure
    psg.generate_normalized_phase_screens(Nps)
    #psg._phaseScreens.shape (10, 1920, 1920)
    # riscalo i phase screens normalizzarti all r0(lambda) che voglio
    r0_vis = 0.2
    psg.rescale_to(r0_vis)
    #converto in metri
    short_exp_screens = psg.get_in_meters()
    #short_exp_screens.shape (10, 1920, 1920)
    wl = 500e-9
    wl_lab = 633e-9
    f_lab = 250e-3
    Dpe_lab = 10.5e-3
    pp_ccd = 4.65e-6
    airy_radius_in_arcsec  = 1.22*wl/Dpe/4.848e-6
    airy_radius_in_px = 1.22*wl_lab/Dpe_lab*f_lab/pp_ccd # diff limited spot on ccd
    pixel_size_on_sky = airy_radius_in_arcsec/airy_radius_in_px
    seeing_in_px = wl / r0_vis / 4.848e-6 / pixel_size_on_sky # seeing resolution in pixel
    
    ccdframe = (1024, 1360)
    cube_ima = np.zeros((Nps, ccdframe[0],ccdframe[1]))
    
    for idx in range(Nps):
        rphi=np.ma.array(short_exp_screens[idx, :frame_size_in_px[0], :], mask=spoc._cmask_obj.mask())
        mirror.set_shape(rphi.flatten(order='C') + tilt)
        sleep(5)
        ima = cam.getFutureFrames(1).toNumpyArray()
        cube_ima[idx] = ima   
    
    long_exp_ima = cube_ima.mean(axis = 0)
    
    roi_le = long_exp_ima[300:700, 200:600]
    rebinned_roi = rebin(roi_le, (40, 40))
    # imshow
    # plot seeing limited along axis
    # compute FWHM seeing limited from rebinned_roi
    # warning: beware of the binning