'''
Created on 30 mag 2022

@author: giuliacarla
'''

import numpy as np
from astropy.io import fits
from matplotlib.pyplot import imshow
import os
from bronte.wfs.subaperture_set import ShSubaperture, ShSubapertureSet
from bronte.wfs.slope_computer import PCSlopeComputer

mcao_root_dir = '/Volumes/GoogleDrive-100122613177551791357/Drive condivisi/LabMCAO'

def test_frame_fname():
    return os.path.join(mcao_root_dir, 'frame_test', 'test_frame_t_0.6_ms.fits')

def main220520_create_test_image(save=False):
    dir_path = os.path.join(mcao_root_dir, 'frame_test')
    data100 = fits.getdata(
        os.path.join(dir_path, '100_frames_t_0.6_ms.fits'))
    data0_cut = data100[920:1410, 825:1315, 0]
    imshow(data0_cut, origin='lower')
    if save == True:
        hdu = fits.PrimaryHDU(data0_cut)
        filename = 'test_frame_t_0.6_ms.fits'
        hdu.writeto(dir_path + filename)
        

def main220530_create_subaperture():
    fname = test_frame_fname()
    ima = fits.getdata(fname)
    sa = ShSubaperture.createSubap(ID=0, detector_shape=ima.shape,
                                   subaperture_size=26, bl=(20, 35))
    fluxSubapsFrame = np.zeros(ima.size)
    flattenFrame = ima.flatten()
    px_values = flattenFrame[sa.pixelList()]
    fluxSubapsFrame[sa.pixelList()] = px_values
    fluxSubapsFrame = fluxSubapsFrame.reshape(ima.shape)
    imshow(fluxSubapsFrame, origin='lower')
    return sa, fluxSubapsFrame


def main220607_create_subaperture_set():
    fname = test_frame_fname()
    ima = fits.getdata(fname)
    detector_shape = ima.shape
    subap_size = 26
    n_subap = 18
    ID_list = np.arange(n_subap * n_subap)
    bl_init = [20, 10]
    x = np.array([bl_init[1] + subap_size * i for i in range(n_subap)])
    y = np.array([bl_init[0] + subap_size * i for i in range(n_subap)])
    xgrid = np.repeat(x[np.newaxis, :], n_subap, axis=0).flatten()
    ygrid = np.repeat(y[np.newaxis, :], n_subap, axis=1).flatten()
    bl_list = [np.array([xgrid, ygrid])[:,i] for i in range(len(ID_list))]
    sset = ShSubapertureSet()
    subaset = sset.createMinimalSet(
        ID_list, detector_shape, subap_size, bl_list)
    return ima, subaset 
    

def main220607_compute_slopes():
    ima, subaset = main220607_create_subaperture_set()
    sl_pc = PCSlopeComputer(subaset)
    sl_pc.set_frame(ima)
    imshow(sl_pc.subapertures_pixels_map(), origin='lower')
    return ima, subaset, sl_pc
