import numpy as np

from arte.types.mask import BaseMask, CircularMask
from arte.types.slopes import Slopes
from arte.utils.modal_decomposer import ModalDecomposer
from arte.utils.rebin import rebin

def get_slope2rad():
    pass

class SyntheticWavefrontReconstrutor():
    
    def __init__(self, slope_computer, Nmodes, pupil_radius_in_m = 10.5e-3, slope2rad = 6.23e-3):
        
        self._sc = slope_computer
        self._n_pix_subap = slope_computer._subapSizeInPx
        self.pupil_radius = pupil_radius_in_m
        self._slope_unit_2_rad = slope2rad
        
        self._Nmodes = Nmodes
        self._md = ModalDecomposer(Nmodes)
        self._subap_mask, self._zernike_mask = self._compute_masks()
        
    def _compute_masks(self):
        
        sub_id_map = self._sc.subapertures_id_map()
        l = np.where(sub_id_map.sum(axis=0))[0].min()
        r = np.where(sub_id_map.sum(axis=0))[0].max()
        b = np.where(sub_id_map.sum(axis=1))[0].min()
        t = np.where(sub_id_map.sum(axis=1))[0].max()
        n_subap_l_r = int((r-l+1)/self._n_pix_subap)
        n_subap_b_t = int((t-b+1)/self._n_pix_subap)
        valid_sub_set = rebin(sub_id_map[b:t+1, l:r+1], (n_subap_b_t, n_subap_l_r))
        valid_sub_set[valid_sub_set > 0] = 1
        subap_mask = (1-valid_sub_set).astype(bool)

        # create Circular Mask to be used as Zernike unitary circle
        zernike_mask = CircularMask((n_subap_b_t, n_subap_l_r))
        return subap_mask, zernike_mask
    
    def compute_zernike_coefficients(self):
        # create Slopes object in rad
        sl = Slopes(self._sc.slopes()[:, 0]*self._slope_unit_2_rad * self.pupil_radius,
                    self._sc.slopes()[:, 1] *
                    self._slope_unit_2_rad * self.pupil_radius,
                    self._subap_mask)

        # use modal decomposer
        zc = self._md.measureZernikeCoefficientsFromSlopes(
            sl, self._zernike_mask, BaseMask(self._subap_mask))
        return zc
 
    @property
    def get_masks(self):
        return self._subap_mask, self._zernike_mask