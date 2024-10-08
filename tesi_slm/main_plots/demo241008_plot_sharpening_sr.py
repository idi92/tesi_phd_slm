import numpy as np
import matplotlib.pyplot as plt
from tesi_slm.utils import my_tools
from arte.types.mask import CircularMask


def sharped_and_unsharped_psf():
    fpath = "C:\\Users\\labot\\Desktop\\misure_tesi_slm\\non_common_path_abs\\230911\\"
    fsharp_c2 = fpath + "230911ima_cleansharppsf_c2_m10umrms_texp1.0ms_30times_v0.fits"
    #funsharp_c2 = fpath + "230912ima_unsharpedpsf_c2_m10umrms_texp1.0ms_v0.fits"
    #fsharp = fpath + "230911ima_cleansharppsf_flat_texp1.0ms_30times_v0.fits"
    funsharp = fpath+ "230911ima_unsharppsf_flat_texp1.0ms_v0.fits"
    
    h, sharp_psf = my_tools.open_fits_file(fsharp_c2)
    h, unsharp_psf = my_tools.open_fits_file(funsharp)
    #h, s_psf_flat = my_tools.open_fits_file(fsharp)
    #h, u_psf_flat = my_tools.open_fits_file(funsharp) 
    
    s_ima = sharp_psf[0].data
    s_par = sharp_psf[1].data
    s_err = sharp_psf[2].data
    
    u_ima = unsharp_psf[0].data
    u_par = unsharp_psf[1].data
    u_err = unsharp_psf[2].data
    
    u_ima[u_ima<0] = 0
    s_ima[s_ima<0] = 0
    
    y,x = my_tools.get_index_from_image(s_ima)
    cut_s_ima = my_tools.cut_image_around_coord(s_ima, y, x, 30)
    
    y,x = my_tools.get_index_from_image(u_ima)
    cut_u_ima = my_tools.cut_image_around_coord(u_ima, y, x, 30)
    
    # print(fname)
    # print(s_par)
    # print(s_err)
    #

    # plt.figure()
    # plt.clf()
    # plt.imshow(s_ima, cmap='jet')
    # plt.colorbar(label='ADU')
    #
    # plt.figure()
    # plt.clf()
    # plt.imshow(u_ima, cmap='jet')
    # plt.colorbar(label='ADU')
    
    plt.figure()
    plt.clf()
    #cut_ima[cut_ima<=0] = 1e-11
    plt.imshow(np.log10(cut_s_ima + 10), cmap='jet')
    plt.colorbar(label='Log scale')
    
    plt.figure()
    plt.clf()
    #cut_ima[cut_ima<=0] = 1e-11
    plt.imshow(np.log10(cut_u_ima + 10), cmap='jet')
    plt.colorbar(label='Log scale')
    
    
    RAD2ARCSEC=180/np.pi*3600
    pupil_diameter = 2*571*9.2e-6
    wavelength = 633e-9 # 1310e-9
    
    ccd_pixel_in_meter = 4.65e-6
    #f_number =  23
    telescope_focal_length = 250e-3
    # so the size of the pixel in arcsec is 
    pixel_scale_in_arcsec =ccd_pixel_in_meter / telescope_focal_length * RAD2ARCSEC 

    # the DL size (=lambda/D) in units of arcsec or pixels are therefore
    dl_size_in_arcsec = wavelength / pupil_diameter * RAD2ARCSEC 
    dl_size_in_pixels = dl_size_in_arcsec / pixel_scale_in_arcsec
    
    print("GC prosilica pixel scale: %g arcsec/pixel" % pixel_scale_in_arcsec)
    print("DL PSF size: %g arcsec" % dl_size_in_arcsec)
    print("DL PSF size: %g pixels" % dl_size_in_pixels)
    
    
    Npix = 200
    pupil_radius_in_pix = Npix/2
    #obstruction_radius_in_pix = 0.33*pupil_radius_in_pix
    dl_psf, dl_psf_scale_in_arcsec = compute_psf_dl(Npix, wavelength, pupil_diameter, pixel_scale_in_arcsec)
    print("Computed DL PSF scale %g" % dl_psf_scale_in_arcsec)
    
    
    total_dl_flux = dl_psf.sum()
    
    total_meas_flux_s = cut_s_ima.sum()
    dl_psf_norm_s = dl_psf * total_meas_flux_s/total_dl_flux
    cc=(dl_psf_norm_s.shape[0]) // 2
    dl_psf_norm_roi_s = dl_psf_norm_s[cc-30:cc+30, cc-30:cc+30]
    
    
    total_meas_flux_u = cut_u_ima.sum()
    dl_psf_norm_u = dl_psf * total_meas_flux_u/total_dl_flux
    cc=(dl_psf_norm_u.shape[0]) // 2
    dl_psf_norm_roi_u = dl_psf_norm_u[cc-30:cc+30, cc-30:cc+30]
    
    v_min = 0
    v_max = np.log10(dl_psf_norm_roi_s.max())
    
    fig ,axs = plt.subplots(1,3,sharex=True,sharey=True)
    im0 = axs[0].imshow(np.log10(np.clip(dl_psf_norm_roi_s,0,None)+1),vmin=v_min, vmax=v_max)
    im1 = axs[1].imshow(np.log10(np.clip(cut_u_ima,0,None)+1),vmin=v_min, vmax=v_max)
    im2 = axs[2].imshow(np.log10(np.clip(cut_s_ima,0,None)+1),vmin=v_min, vmax=v_max)
    fig.colorbar(im2, ax = axs[2])
    
    plt.figure()
    plt.imshow(np.log10(np.clip(dl_psf_norm_roi_s,0,None)+1))
    plt.colorbar()
    plt.figure()
    plt.imshow(np.log10(np.clip(cut_u_ima,0,None)+1))
    plt.colorbar()
    
    print("Max(DL PSF normalized) %g" % dl_psf_norm_roi_s.max())
    print("Max(Measured PSF) %g" % cut_s_ima.max())
    strehl_ratio_s = cut_s_ima.max() / dl_psf_norm_roi_s.max()
    strehl_ratio_u = cut_u_ima.max() / dl_psf_norm_roi_u.max()
    print("SR(sharp): %g" % strehl_ratio_s)
    print("SR(unsharp): %g" % strehl_ratio_u)

def compute_psf_dl(Npix, wavelength, pupil_diameter, pixel_scale_in_arcsec):
    pupil = np.zeros((Npix,Npix))
    pupil_radius_in_pix = Npix/2
    

    pupil_mask_obj = CircularMask(
        frameShape=pupil.shape,
        maskRadius=pupil_radius_in_pix,
        )

    pupil_mask = pupil_mask_obj.mask()
    phase = np.ma.array(pupil, mask = pupil_mask)

    # computing transmitted electric field
    Ut = 1 * np.exp(1j * phase)
    Ut.fill_value = 0
    Ut.data[Ut.mask == True] = 0

    RAD2ARCSEC=180/np.pi*3600
    # padding transmitted electric field to match resulting px scale with the instrument pixel scale
    Npad = wavelength / pupil_diameter  * RAD2ARCSEC /  pixel_scale_in_arcsec
    print("Pupil padding %g" % Npad)
    padded_frame_size = int(np.round(Npix * Npad))
    padded_Ut = np.zeros((padded_frame_size, padded_frame_size), dtype=complex)
    padded_Ut[0 : Ut.shape[0], 0 : Ut.shape[1]] = Ut   

    plt.figure()
    plt.imshow(np.abs(padded_Ut))
    plt.colorbar()


    #computing psf
    dl_psf = np.abs(np.fft.fftshift(np.fft.fft2(padded_Ut)))**2
    plt.figure()
    plt.imshow(dl_psf)
    plt.colorbar()

    dl_psf_scale_in_arcsec = wavelength / pupil_diameter / Npad * RAD2ARCSEC

    return dl_psf, dl_psf_scale_in_arcsec
    
    
    