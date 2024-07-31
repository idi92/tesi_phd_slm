import numpy as np
from tesi_slm.utils import fits_io
from tesi_slm.calib.wfs import shwfs_tilt_calibration
import matplotlib.pyplot as plt


from arte.types.mask import BaseMask, CircularMask
from arte.types.slopes import Slopes
from arte.utils.modal_decomposer import ModalDecomposer
from arte.utils.rebin import rebin


fname_sh = '/Users/lbusoni/Library/CloudStorage/GoogleDrive-lorenzo.busoni@inaf.it/.shortcut-targets-by-id/1SPpwbxlHyuuXmzaajup9lXpg_qSHjBX4/phd_slm_edo/misure_tesi_slm/shwfs_calibration/240717_tilt_linearity_on_subapertures/data/red_data/240730shwfs_tilt_calib_adjX.fits'


def update_subaps_threshold(subaps, threshold):
    """
    This should be in SubapertureSet
    """
    for i in subaps.values():
        i.setFixThreshold(threshold)


def main(fname_sh, fixThreshold=0):


    shima, shhdr, shrevolution = fits_io.load(fname_sh)
    wf_ref = shima[:, :, 0]
    sgi = shwfs_tilt_calibration.main(wf_ref)
    update_subaps_threshold(sgi._subaps, fixThreshold)
    slv = []
    valid_meas_idx = range(0, 3)
    for i in valid_meas_idx:
        sgi._sc.set_frame(shima[:, :, i])
        sl = sgi._sc.slopes()
        slv.append(sl)
    slv = np.array(slv)
    sl_ref = slv[0, :, :]

    plt.figure()
    for i in valid_meas_idx:
        plt.plot(slv[i, :, 0]-sl_ref[:, 0])

    return sgi, slv, sl_ref


class WavefrontReconstructor:
    def __init__(self, slope_computer):
        self._sc = slope_computer
        self._n_pix_subap = 26
        self.pupil_radius = 5.25e-3

        # geometric factor for 10.5mm pupil and lab setup at 240731
        self._slope_unit_2_rad = 6.23e-3

        # compute 100 Zernikes
        self._md = ModalDecomposer(50)

        self._subap_mask, self._zernike_mask = self._compute_masks()

    def _compute_masks(self):
        dd = self._sc.subapertures_id_map()
        l = np.where(dd.sum(axis=0))[0].min()
        r = np.where(dd.sum(axis=0))[0].max()
        b = np.where(dd.sum(axis=1))[0].min()
        t = np.where(dd.sum(axis=1))[0].max()
        n_subap_l_r = int((r-l+1)/self._n_pix_subap)
        n_subap_b_t = int((t-b+1)/self._n_pix_subap)
        dd = rebin(dd[b:t+1, l:r+1], (n_subap_b_t, n_subap_l_r))
        dd[dd > 0] = 1
        maska = (1-dd).astype(bool)

        # create Circular Mask to be used as Zernike unitary circle
        mask = CircularMask((n_subap_b_t, n_subap_l_r))
        return maska, mask

    def _compute_zernike_coefficients(self):
        # create Slopes object in rad
        sl = Slopes(self._sc.slopes()[:, 0]*self._slope_unit_2_rad * self.pupil_radius,
                    self._sc.slopes()[:, 1] *
                    self._slope_unit_2_rad * self.pupil_radius,
                    self._subap_mask)

        # use modal decomposer
        zc = self._md.measureZernikeCoefficientsFromSlopes(
            sl, self._zernike_mask, BaseMask(self._subap_mask))
        return zc


def main_reconstructed_tt(fname_sh, fixThreshold=100):

    shima, shhdr, shrevolution = fits_io.load(fname_sh)
    wf_ref = shima[:, :, 0]
    sgi = shwfs_tilt_calibration.main(wf_ref)
    update_subaps_threshold(sgi._subaps, fixThreshold)

    wr = WavefrontReconstructor(sgi._sc)
    sgi._sc.set_frame(shima[:, :, 0])
    zc_0 = wr._compute_zernike_coefficients()
    sgi._sc.set_frame(shima[:, :, 1])
    zc_1 = wr._compute_zernike_coefficients()

    plt.figure()
    plt.plot(zc_1.zernikeIndexes(), (zc_1.toNumpyArray() -
             zc_0.toNumpyArray()), '.-')
    plt.grid(True)
    plt.xlim(2, 20)
    plt.title(
        "reconstructed wavefront for 0.125rev on Thorlabs tilt mount")
    plt.xlabel('Zernike mode')
    plt.ylabel('Modal amplitude wf rms [m]')
