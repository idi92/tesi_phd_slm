import numpy as np 
from tesi_slm.my_tools import open_fits_file, get_circular_mask_obj, get_wf_as_zerike_combo
import matplotlib.pyplot as plt
from astropy.io import fits 

def lorenzo_example():
    
    cmask_obj = get_circular_mask_obj((571,875), 571, (1152, 1920))
    fname = "C:\\Users\\labot\\Desktop\\misure_tesi_slm\\SHWFS meas\\231019\\231019shwfs_ima_Z2_1000.fits"
    slm_hdr, slm_data = open_fits_file(fname)
    
    sh_ima_slm = slm_data[0].data
    #sharpening coeff on tilted psf -10um rms
    best_coeff = slm_data[2].data
    # defocusing ghost 1e-6 m rms
    bias4 = slm_data[3].data
    # tilt applied wrt the refference coeffs
    coeff2apply = slm_data[4].data
    
    plt.figure()
    plt.clf()
    plt.imshow(sh_ima_slm, cmap = 'jet')
    plt.colorbar()
    
    plt.figure()
    plt.clf()
    plt.imshow(sh_ima_slm[1595:1818, 860:1205], cmap= 'jet')
    plt.colorbar()
    
    coeff_wf1 = [1e-6, 0, 1e-6]
    coeff_wf2 = [0]
    
    wf1 = get_wf_as_zerike_combo(cmask_obj, coeff_wf1)
    wf2 = get_wf_as_zerike_combo(cmask_obj, coeff_wf2)
    
    e1=1*np.exp(1j*2*np.pi*wf1/633e-9)
    e2=1*np.exp(1j*2*np.pi*wf2/633e-9)
    
    eta = [0.95, 0.05]
    
    interf = np.abs(eta[0]*e1 + eta[1]*e2)**2
    plt.figure()
    plt.clf()
    plt.imshow(interf)
    plt.colorbar() 
    
def save_interferograms(fname, interf, coeff_wf1, coeff_wf2, eta):
    fits.writeto(fname , interf)
    fits.append(fname, coeff_wf1)
    fits.append(fname, coeff_wf2)
    fits.append(fname, eta)
    