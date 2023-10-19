
from tesi_slm import sharp_psf_on_camera
import pysilico
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

def main_lorenzo():
    
    cam, mirror = sharp_psf_on_camera.create_devices()
    spoc = sharp_psf_on_camera.SharpPsfOnCamera(cam, mirror)
    spoc.change_circular_mask((571,875), 571)
    
    shcam = pysilico.camera('localhost',7110)
    shcam.setExposureTime(2)
    
    bc28 = np.array([-1.0e-5,  0.0,  5.5e-8,  1.5e-8, 1.5e-8, -4.0e-8,  1.0e-8,\
        1.5e-8, 3.0e-8,  2.5e-8, -5.0e-9,  0,-5.0e-9,  5.0e-9, -5.0e-9,  5.0e-9, 5.0e-9,\
        0,  0,  0,  0,  1.0e-08,  0,  0,  0,  0,  0])
    
    bias4 = np.zeros(27)
    bias4[2] = 1e-6
    
    spoc.write_zernike_on_slm(bc28+bias4)
    ima_r = shcam.getFutureFrames(1,3).toNumpyArray()
    
    wf_to_apply=np.zeros(27)
    wf_to_apply[1]=100e-9
    spoc.write_zernike_on_slm(bc28+bias4+wf_to_apply)
    ima_Z3_100 = shcam.getFutureFrames(1,3).toNumpyArray()
    plt.clf();plt.imshow(ima_Z3_100-ima_r); plt.colorbar()
    
    spoc.write_zernike_on_slm(bc28 + bias4)
    ima_0 = shcam.getFutureFrames(1,3).toNumpyArray()
    plt.clf();plt.imshow(ima_0-ima_r); plt.colorbar()
    
    wf_to_apply=np.zeros(27); wf_to_apply[2]=100e-9
    spoc.write_zernike_on_slm(bc28+bias4+wf_to_apply)
    
    wf_to_apply=np.zeros(27); wf_to_apply[1]=1000e-9
    spoc.write_zernike_on_slm(bc28+bias4+wf_to_apply)
    ima_Z3_1000 = shcam.getFutureFrames(1,3).toNumpyArray()
    plt.clf();plt.imshow(ima_Z3_1000-ima_r); plt.colorbar()
    ima_Z4_100 = shcam.getFutureFrames(1,3).toNumpyArray()
    plt.clf();plt.imshow(ima_Z4_100-ima_r); plt.colorbar()
    
    wf_to_apply=np.zeros(27); wf_to_apply[0]=2000e-9
    spoc.write_zernike_on_slm(bc28+bias4+wf_to_apply)
    ima_Z2_2000 = shcam.getFutureFrames(1,3).toNumpyArray()
    plt.clf();plt.imshow(ima_Z2_2000); plt.colorbar()
    
def save_shwfs_ima(fname, ima, best_coeff, bias4, coef2apply, texp):
    hdr = fits.Header()
    hdr['T_EX_MS'] = texp
    fits.writeto(fname , ima, hdr)
    fits.append(fname, best_coeff + bias4 + coef2apply)
    fits.append(fname, best_coeff)
    fits.append(fname, bias4)
    fits.append(fname, coef2apply)