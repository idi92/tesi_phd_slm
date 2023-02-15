'''
Created on 7 Feb 2023

@author: labot
'''
import numpy as np
import pysilico


def main20230207_psf_fluctuation(exp_time_ms=2):
    cam = pysilico.camera('localhost', 7100)
    cam.setExposureTime(exp_time_ms)
    n_imas = 100
    imas =  cam.getFutureFrames(n_imas).toNumpyArray()
    max_idcs_list = []
    for i in range(n_imas):
        id_max = np.unravel_index(imas[:,:,i].argmax(), imas[:,:,i].shape)
        max_idcs_list.append(id_max)
    sy, sx = np.array(max_idcs_list).std(axis=0)
    return imas, max_idcs_list, sx, sy
        
    
def optimize_ncpa(cam, dm_o_pippo, coeffs):
    pippo = dm_o_pippo
    cmask = pippo.get_default_circular_mask()
    maxi = np.zeros(coeffs.shape)
    for i, coeff in enumerate(coeffs):
        pippo.write_zernike_on_slm(cmask,[0,0, -160e-9, 0e-9, coeff],True)
        fr=cam.getFutureFrames(30)
        maxi[i]=fr.toNumpyArray().max(axis=(0,1)).mean()
    return maxi