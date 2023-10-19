import numpy as np
import matplotlib.pyplot as plt
from tesi_slm import my_tools
from astropy.io import fits

def sharped_and_unsharped_psf(fname):
    #fpath = "C:\\Users\\labot\\Desktop\\misure_tesi_slm\\non_common_path_abs\\230911\\"
    #fsharp_c2 = fpath + "230911ima_cleansharppsf_c2_m10umrms_texp1.0ms_30times_v0.fits"
    #funsharp_c2 = fpath + "230912ima_unsharpedpsf_c2_m10umrms_texp1.0ms_v0.fits"
    #fsharp = fpath + "230911ima_cleansharppsf_flat_texp1.0ms_30times_v0.fits"
    #funsharp = fpath+ "230911ima_unsharppsf_flat_texp1.0ms_v0.fits"
    
    h, psf = my_tools.open_fits_file(fname)
    #h, u_psf_tilt = my_tools.open_fits_file(funsharp_c2)
    #h, s_psf_flat = my_tools.open_fits_file(fsharp)
    #h, u_psf_flat = my_tools.open_fits_file(funsharp) 
    
    ima = psf[0].data
    par = psf[1].data
    err = psf[2].data
    y,x = my_tools.get_index_from_image(ima)
    cut_ima = my_tools.cut_image_around_coord(ima, y, x, 15)
    
    print(fname)
    print(par)
    print(err)
    
    plt.figure()
    plt.clf()
    plt.imshow(cut_ima, cmap='jet')
    plt.colorbar(label='ADU')
    plt.figure()
    plt.clf()
    #cut_ima[cut_ima<=0] = 1e-11
    plt.imshow(np.log(cut_ima + 10), cmap='jet')
    plt.colorbar(label='Log scale')

def difference_between_plico_and_blink():
    from tesi_slm import demo230407_difference_blink_spoc
    fname_blink = "C:\\Users\\labot\\Desktop\\misure_tesi_slm\\230403\\z6_c6_2um_r569_y550_x853_bmp_blink.fits.bmp"
    fname_spoc = "C:\\Users\\labot\\Desktop\\misure_tesi_slm\\230403\\z6_c6_2um_r569_y550_x853_bmp.fits"
    demo230407_difference_blink_spoc.main(fname_blink, fname_spoc)
    
def tilt_linearity():
    from tesi_slm import tilt_linearity_analyzer_new
    
    fname_z2 = "C:\\Users\\labot\\Desktop\\misure_tesi_slm\\230414\\again\\230414tpm_red_z2_v4.fits"
    fname_z3 = "C:\\Users\\labot\\Desktop\\misure_tesi_slm\\230414\\again\\230414tpm_red_z3_v4.fits"
    
    tpa2 = tilt_linearity_analyzer_new.TiltedPsfAnalyzer(fname_z2)
    tpa2.compute_tilted_psf_desplacement()
    tpa2.execite_linfit_along1axis()
    tpa3 = tilt_linearity_analyzer_new.TiltedPsfAnalyzer(fname_z3)
    tpa3.compute_tilted_psf_desplacement()
    tpa3.execite_linfit_along1axis()
    
def sharpening_psf_histo():
    # sharpening from j=4 to j=11
    # misure ripetute 30 volte
    fpath = "C:\\Users\\labot\\Desktop\\misure_tesi_slm\\non_common_path_abs\\230911\\"
    fnflat30 = fpath + "230911spoc_coef_matrix_flat_texp1.0ms_30times_v0.fits"
    fntiltm5um = fpath + "230912spoc_coef_matrix_c2_m5umrms_texp1.0ms_v0.fits"
    fntiltm10um = fpath + "230911spoc_coef_matrix_c2_m10umrms_texp1.0ms_30times_v0.fits"
    fntiltm20um = fpath + "230911spoc_coef_matrix_c2_m20umrms_texp1.0ms_30times_v0.fits"
    
    hh , dd = my_tools.open_fits_file(fnflat30)
    coeff_flat = dd[0].data.mean(axis=1)
    err_coeff_flat = dd[0].data.std(axis=1)
    
    hh , dd = my_tools.open_fits_file(fntiltm5um)
    coeff_z2_m5um = dd[0].data.mean(axis=1)
    err_coeff_z2_m5um = dd[0].data.std(axis=1)
    
    hh , dd = my_tools.open_fits_file(fntiltm10um)
    coeff_z2_m10um = dd[0].data.mean(axis=1)
    err_coeff_z2_m10um = dd[0].data.std(axis=1)
    
    hh , dd = my_tools.open_fits_file(fntiltm20um)
    coeff_z2_m20um = dd[0].data.mean(axis=1)
    err_coeff_z2_m20um = dd[0].data.std(axis=1)
    
    j_index = np.arange(2,11+1)

    plt.figure()
    plt.clf()
    nm = 1e-9
    dj = 0.2
    dw = 0.4
    
    plt.bar(j_index, coeff_flat/nm, width = -dw ,align='edge', color='r',label='c2 = 0 um rms')
    plt.bar(j_index,coeff_z2_m10um/nm, width=dw ,align='edge',color='g',label='c2 = -10 um rms')
    
    plt.errorbar(j_index-dj, coeff_flat/nm, err_coeff_flat/nm, fmt='ko', ecolor='k',linestyle='')
    plt.errorbar(j_index+dj, coeff_z2_m10um/nm, err_coeff_z2_m10um/nm, fmt='ko', ecolor ='k', linestyle='')
    
    plt.xlabel('j index')
    plt.xticks(j_index)
    plt.ylabel('$c_j$'+'' '[nm rms]')
    plt.xlim(4-0.5,11+0.5)
    plt.ylim(-75,75)
    plt.legend(loc='best')
    plt.grid(ls='--',alpha = 0.3)
    
    plt.figure()
    plt.clf()
    dj = 0.2*0.5
    dw = 0.2
    plt.bar(j_index-3*dj, coeff_flat/nm, width = -dw ,align='center', color='r',label='c2 = 0 um rms')
    plt.bar(j_index-dj,coeff_z2_m5um/nm, width=-dw ,align='center',color='c',label='c2 = -5 um rms')
    plt.bar(j_index+dj,coeff_z2_m10um/nm, width=dw ,align='center',color='g',label='c2 = -10 um rms')
    plt.bar(j_index+3*dj,coeff_z2_m20um/nm, width=dw ,align='center',color='m',label='c2 = -20 um rms')
    
    plt.errorbar(j_index-3*dj, coeff_flat/nm, err_coeff_flat/nm, fmt='ko', ecolor='k',linestyle='')
    plt.errorbar(j_index-dj, coeff_z2_m5um/nm, err_coeff_z2_m5um/nm, fmt='ko', ecolor ='k', linestyle='')
    plt.errorbar(j_index+dj, coeff_z2_m10um/nm, err_coeff_z2_m10um/nm, fmt='ko', ecolor ='k', linestyle='')
    plt.errorbar(j_index+3*dj, coeff_z2_m20um/nm, err_coeff_z2_m20um/nm, fmt='ko', ecolor ='k', linestyle='')
    
    plt.xlabel('j index')
    plt.xticks(j_index)
    plt.ylabel('$c_j$'+'' '[nm rms]')
    plt.xlim(4-0.5,11+0.5)
    plt.ylim(-75,75)
    plt.legend(loc='best')
    plt.grid(ls='--',alpha = 0.3)

def ghost_lp_rot():
    # intensity measurements as a function of the LP rotation angle
    from tesi_slm import ghost_measurer
    fpath = "C:\\Users\\labot\\Desktop\\misure_tesi_slm\\230315\\"
    fname_tag='230315gm_ang'
    angles = [300, 320, 330, 336, 340, 342, 344, 346, 348, 350, 356, 360, 370]
    agr = ghost_measurer.AnalyzeGhostRatio(angles, fpath,fname_tag)
    agr.show_ratio()
    # measurements around 340 deg
    fpath = "C:\\Users\\labot\\Desktop\\misure_tesi_slm\\230317\\"
    angles = [3380, 3381,3382,3383,3384,3385,3400,3401,3402,3403,3404,3405,3406,3420]
    fname_tag='230317gm_ang'
    agr = ghost_measurer.AnalyzeGhostRatio(angles, fpath, fname_tag)
    agr._rot_angle =np.array([338,338.35,338.7,339.05,339.4,339.75,
                              340.1,340.45,340.8,341.15,341.5,341.85,342.2,342.55]) 
    agr.show_ratio()

def diffraction_eff():
    
    fname_spa = "C:\\Users\\labot\\Desktop\\misure_tesi_slm\\230414\\again\\230414dem_z2_v0.fits"
    fname_thor = "C:\\Users\\labot\\Desktop\\misure_tesi_slm\\230426\\230426dem_z2_v0_thorlrand.fits"
    h_spa, d_spa = my_tools.open_fits_file(fname_spa)
    Im_spa = d_spa[0].data
    errIm_spa = d_spa[1].data
    Ig_spa = d_spa[2].data
    errIg_spa = d_spa[3].data
    c_span = d_spa[4].data
    init_coeff = d_spa[5].data
    
    h_thor, d_thor = my_tools.open_fits_file(fname_thor)
    Im_thor = d_thor[0].data
    errIm_thor = d_thor[1].data
    Ig_thor = d_thor[2].data
    errIg_thor = d_thor[3].data
    c_span = d_thor[4].data
    init_coeff = d_thor[5].data
    
    plt.subplots(2,1,sharex=True)
    plt.subplot(2,1,1)
    plt.plot(c_span, Ig_spa, 'bo', label = 'SP117A')
    plt.plot(c_span, Ig_thor, 'bx', label = 'THORLABS', alpha=0.5)
    plt.errorbar(c_span, Ig_spa, errIg_spa, fmt='.b', ecolor ='b', linestyle='')
    plt.errorbar(c_span, Ig_thor, errIg_thor, fmt='.b', ecolor ='b', linestyle='',alpha=0.5)
    plt.xlabel('c [m]')
    plt.ylabel('$<I_{ghost}> / <I_{flat}>$')
    plt.legend(loc='best')
    plt.grid(ls ='--', alpha = 0.5)
    plt.subplot(2,1,2)
    plt.plot(c_span, Im_spa, 'ro', label = 'SP117A')
    plt.plot(c_span, Im_thor, 'rx', label = 'THORLABS', alpha=0.5)
    plt.errorbar(c_span, Im_spa, errIm_spa, fmt='.r', ecolor ='r', linestyle='')
    plt.errorbar(c_span, Im_thor, errIm_thor, fmt='.r', ecolor ='r', linestyle='',alpha=0.5)
    plt.xlabel('c [m]')
    plt.ylabel('$<I_{modulated}> / <I_{flat}>$')
    plt.legend(loc='best')
    plt.grid(ls ='--', alpha = 0.5)
    
    #avoid c2 = 0 um rms
    mmask = np.zeros(len(c_span))
    mmask[4] = 1
    c_span_m = np.ma.array(c_span, mask = mmask)
    Ig_spa_m = np.ma.array(Ig_spa, mask = mmask)
    errIg_spa_m =  np.ma.array(errIm_spa, mask = mmask)
    Im_spa_m = np.ma.array(Im_spa, mask = mmask)
    errIm_spa_m =  np.ma.array(errIm_spa, mask = mmask)
    plt.subplots(2,1,sharex=True)
    plt.subplot(2,1,1)
    plt.plot(c_span_m, Ig_spa_m, 'bo', label = 'ghost')
    plt.errorbar(c_span_m, Ig_spa_m, errIg_spa_m, fmt='.b', ecolor ='b', linestyle='')
    #plt.xlabel('$c [m] rms$')
    plt.ylabel('$<I_{ghost}> / <I_{flat}>$')
    plt.legend(loc='best')
    plt.grid(ls ='--', alpha = 0.5)
    #plt.ylim(0.03,0.075)
    plt.subplot(2,1,2)
    plt.plot(c_span_m, Im_spa_m, 'ro', label = 'mod')
    plt.errorbar(c_span_m, Im_spa_m, errIm_spa_m, fmt='.r', ecolor ='r', linestyle='')
    #plt.ylim(0.85,1.05)
    plt.xlabel('$c$'+'  [m] rms')
    plt.ylabel('$<I_{modulated}> / <I_{flat}>$')
    plt.legend(loc='best')
    plt.grid(ls ='--', alpha = 0.5)
    
    
def get_best_coeff_from_psf_sharpening():
    fpath = "C:\\Users\\labot\\Desktop\\misure_tesi_slm\\non_common_path_abs\\230911\\"
    #j from 4 to 28
    fflatv0 = "230918spoc_coef_matrix_flat_texp1.0ms_10times_v0.fits"
    fflatv1 = "230918spoc_coef_matrix_flat_texp1.0ms_10times_v1.fits"
    fflatv2 = "230918spoc_coef_matrix_flat_texp1.0ms_10times_v2.fits"
    fflatv3 = "230919spoc_coef_matrix_flat_texp1.0ms_10times_v3.fits"
    fflatv4 = "230919spoc_coef_matrix_flat_texp1.0ms_10times_v4.fits"
    ftiltv0 = "230920spoc_coef_matrix_c2p10umrms_texp1.0ms_10times_v0.fits"
    ftiltv1 = "230920spoc_coef_matrix_c2p10umrms_texp1.0ms_10times_v1.fits"
    ftiltv2 = "230921spoc_coef_matrix_c2p10umrms_texp1.0ms_10times_v2.fits"
    ftiltv3 = "230921spoc_coef_matrix_c2p10umrms_texp1.0ms_10times_v3.fits"
    ftiltv4 = "230922spoc_coef_matrix_c2p10umrms_texp1.0ms_10times_v4.fits"
    # j from 4 to 11
    ftiltm20 = "230911spoc_coef_matrix_c2_m20umrms_texp1.0ms_30times_v0.fits"
    ftiltm10  ="230911spoc_coef_matrix_c2_m10umrms_texp1.0ms_30times_v0.fits"
    fflat11 = "230911spoc_coef_matrix_flat_texp1.0ms_30times_v0.fits"
    
    Ntimes = 50
    for idx in np.arange(Ntimes):
        hh , dd = my_tools.open_fits_file(fname)
        coeff_flat = dd[0].data.mean(axis=1)
    
    return coeff_flat

def get_flat_best_coeff():
    fpath = "C:\\Users\\labot\\Desktop\\misure_tesi_slm\\non_common_path_abs\\230911\\"
    fflatv4 = "230919spoc_coef_matrix_flat_texp1.0ms_10times_v4.fits"
    ftiltm10  ="230911spoc_coef_matrix_c2_m10umrms_texp1.0ms_30times_v0.fits"
    fflat11 = "230911spoc_coef_matrix_flat_texp1.0ms_30times_v0.fits"
    fname = fpath + fflat11
    hh , dd = my_tools.open_fits_file(fname)
    coeff_flat = dd[0].data.mean(axis=1)
    return coeff_flat