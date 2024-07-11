import numpy as np 
import matplotlib.pyplot as plt
from arte.atmo.phase_screen_generator import PhaseScreenGenerator
from tesi_slm.phase_screen_on_slm.display_atmo_on_slm import SeeingLimitedAnalyser
from tesi_slm.my_tools import convert_opd2wrapped_phase, get_circular_mask, open_fits_file

def show_plots_phase_screens_on_slm():
    
    fpath =  "C:\\Users\\labot\\Desktop\\misure_tesi_slm\\display_phase_screens\\kolmogorov_atmo\\class8m\\"
    fname_nps = fpath + "231005normalized_phase_screen_cube_8m_571px_os100m_n100.fits"
    
    nps = PhaseScreenGenerator.load_normalized_phase_screens(fname_nps)
    r0 = 0.2
    nps.rescale_to(r0)
    ps_cube = nps.get_in_meters()
    
    plt.figure()
    plt.clf()
    plt.imshow(ps_cube[0, :1152,:])#, cmap = 'jet')
    #plt.colorbar()
    
    plt.figure()
    plt.clf()
    plt.imshow(ps_cube[1, :1152,:])#, cmap = 'jet')
    #plt.colorbar()
    
    plt.figure()
    plt.clf()
    plt.imshow(ps_cube[2, :1152,:])#, cmap = 'jet')
    #plt.colorbar()
    
    #gray commands
    cmask = get_circular_mask()
    map0 = np.ma.array(data = ps_cube[0, :1152,:], mask = cmask)
    map1 = np.ma.array(data = ps_cube[1,:1152,:], mask = cmask)
    map2 = np.ma.array(data = ps_cube[2,:1152,:], mask = cmask)
     
    plt.figure()
    plt.clf()
    plt.imshow(convert_opd2wrapped_phase(map0), cmap = 'gray')
    #plt.colorbar()
    
    plt.figure()
    plt.clf()
    plt.imshow(convert_opd2wrapped_phase(map1), cmap = 'gray')
    #plt.colorbar()
    
    plt.figure()
    plt.clf()
    plt.imshow(convert_opd2wrapped_phase(map2), cmap = 'gray')
    #plt.colorbar()
    
def show_plots_seima8m_on_ccd():
    fpath =  "C:\\Users\\labot\\Desktop\\misure_tesi_slm\\display_phase_screens\\kolmogorov_atmo\\class8m\\"
    fname_seima = fpath + "231024daos_seima_r0vis0.2__Nfr100texp10ms_bias_c2_m10umrms.fits"
    
    head,hdrdat = open_fits_file(fname_seima) 
    seima_cube = hdrdat[0].data
    bias = hdrdat[1].data
    texp = head['T_EX_MS']
    Nframes = head['N_AV_FR']
    r0 = head['R_0']
    
    
    #roi on tilt
    plt.figure()
    plt.clf()
    plt.imshow(seima_cube[0,425:575,825:975], cmap = 'jet')
    plt.colorbar()
    
    plt.figure()
    plt.clf()
    plt.imshow(seima_cube[1, 425:575,825:975], cmap = 'jet')
    plt.colorbar()
    
    plt.figure()
    plt.clf()
    plt.imshow(seima_cube[2, 425:575,825:975], cmap = 'jet')
    plt.colorbar()
    # roi on ghost and tilt
    plt.figure()
    plt.clf()
    plt.imshow(seima_cube[0,400:600,650:1000], cmap = 'jet')
    plt.colorbar()
    
    plt.figure()
    plt.clf()
    plt.imshow(seima_cube[1,400:600,650:1000], cmap = 'jet')
    plt.colorbar()
    
    plt.figure()
    plt.clf()
    plt.imshow(seima_cube[2, 400:600,650:1000], cmap = 'jet')
    plt.colorbar()
    
def show_plot_shwfs_examples(fname_shwfs_ima):
    fpath = "C:\\Users\\labot\\Desktop\\misure_tesi_slm\\SHWFS meas\\231019\\"
    # if j==2:
    #     fname = fpath + "231019shwfs_ima_Z2_1000.fits"
    # if j==3:
    #     fname = fpath + "231019shwfs_ima_Z3_1000.fits"
    # if j==4:
    #     fname = fpath + "231019shwfs_ima_Z4_100.fits"
    # else:
    #     fname = fpath + "231019shwfs_ima_ref_v0.fits"
    
    fname = fpath + fname_shwfs_ima
    hh, dd = open_fits_file(fname)
    shwfs_ima = dd[0].data
    tot_coeff = dd[1].data
    best_coeff = dd[2].data
    bias4 = dd[3].data
    coeff2apply = dd[4].data
    
    
    print("File Name: %s"%fname_shwfs_ima)
    # print("\nfits header:")
    # print(hh)
    print("\ntot coeff:")
    print(tot_coeff)
    print("\nbest_coeff:")
    print(best_coeff)
    print("\ndefocus bias:")
    print(bias4)
    print("\ncoeff2apply:")
    print(coeff2apply)
    
    Imin = shwfs_ima.min()
    Imax = shwfs_ima.max()
    plt.figure()
    plt.clf()
    plt.hist(shwfs_ima[350:1940,400:1910].ravel(),bins=range(int(Imax)), fc='k', ec='k')
    
    cmin=200 # background
    cmax = 1500
    
    plt.figure()
    plt.clf()
    plt.imshow(shwfs_ima[350:1940,400:1910], cmap='Greys_r', clim = (cmin, cmax))
    plt.colorbar(label='ADU')
    plt.title(fname_shwfs_ima)
    plt.figure()
    plt.clf()
    plt.imshow(shwfs_ima[1510:1790,778:1115], cmap = 'Greys_r', clim = (cmin, cmax))#clim = (Imin, 1.75*Imax)) #vmin = Imin, vmax = Imax)#[1000:1250,950:1350]
    plt.colorbar(label='ADU')
    plt.title(fname_shwfs_ima)
    
    