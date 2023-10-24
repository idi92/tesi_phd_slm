import numpy as np
from tesi_slm.sharp_psf_on_camera import create_devices
from tesi_slm.my_tools import get_circular_mask_obj, get_zernike_wf
from tesi_slm.phase_screen_on_slm.display_atmo_on_slm import DisplayAtmOnSlm, SeeingLimitedAnalyser
from tesi_slm.my_tools import reshape_map2vector
import matplotlib.pyplot as plt
from arte.utils.rebin import rebin

def main(fname_nps):
    
    # fpath = "C:\\Users\\labot\\Desktop\\misure_tesi_slm\\display_phase_screens\\kolmogorov_atmo\\class8m\\"
    # fname = "231005normalized_phase_screen_cube_8m_571px_os100m_n100.fits"
    # fname_nps = fpath + fname
    
    cam, mirror = create_devices()
    cmask_obj = get_circular_mask_obj((571,875), 571, (1152,1920))
    daos = DisplayAtmOnSlm(fname_nps, cam, mirror, cmask_obj) 
    
    r0_vis = 0.2
    
    tilt_bias = get_zernike_wf(cmask_obj, 2, -10e-6)
    
    bias = reshape_map2vector(tilt_bias)
    
    mirror.set_shape(bias)
    texp = 10
    seima = daos.get_short_exp_images(r0_vis, texp, 0.5, bias)
    intima = seima.mean(axis=0)
    fsav =  "C:\\Users\\labot\\Desktop\\misure_tesi_slm\\display_phase_screens\\kolmogorov_atmo\\class8m\\231024daos_seima_r0vis0.2_texp10ms_bias_c2_m10umrms.fits"
    daos.save_short_exp_images(fsav)
    
    sla = SeeingLimitedAnalyser(fsav)
    
    seima = sla._short_exp_images
    intima = seima.mean(axis=0)
    roi_le = intima[450:600,825:975]
    rebinned_roi = rebin(roi_le,(25,25))
    plt.figure();plt.clf();plt.imshow(rebinned_roi,cmap='jet');plt.colorbar()
    