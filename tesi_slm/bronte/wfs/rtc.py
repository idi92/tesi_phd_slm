import numpy as np
from arte.types.mask import BaseMask, CircularMask
from arte.types.slopes import Slopes
from arte.utils.rebin import rebin
from arte.types.zernike_coefficients import ZernikeCoefficients


class ScaoRealTimeComputer:

    def __init__(self, wfs_camera, slope_computer, deformable_mirror, modal_decomposer, controller, slm_rasterizer):
        self._wfs_camera = wfs_camera
        self._sc = slope_computer
        self._dm = deformable_mirror
        self._md = modal_decomposer
        self._controller = controller
        self._slm_rasterizer = slm_rasterizer

        self.pupil_radius = 5.25e-3

        # geometric factor for 10.5mm pupil and lab setup at 240731
        self._slope_unit_2_rad = 6.23e-3

        self._subap_mask, self._zernike_mask = self._compute_masks()

    def _compute_masks(self):
        # TODO move the creation of mask in the slope_computer or in the subaperture set
        dd = self._sc.subapertures_id_map()
        l = np.where(dd.sum(axis=0))[0].min()
        r = np.where(dd.sum(axis=0))[0].max()
        b = np.where(dd.sum(axis=1))[0].min()
        t = np.where(dd.sum(axis=1))[0].max()
        n_subap_l_r = int((r-l+1)/self._sc.subaperture_size)
        n_subap_b_t = int((t-b+1)/self._sc.subaperture_size)
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

    def step(self):
        # Acquire frame
        wfs_frame = self._wfs_camera.getFutureFrames(1, 1).toNumpyArray()
        # Use frame
        # TODO set background_frame for subtraction
        self._sc.upload_raw_frame(wfs_frame)
        # reconstruct Zernike coefficients
        zc = self._compute_zernike_coefficients()
        # temporal_filter
        zc_filtered = self._controller.process_delta_command(zc.toNumpyArray())
        # convert modal amplitudes in SLM shape
        slm_raster = self._slm_rasterizer.zernike_coefficients_to_raster(
            ZernikeCoefficients.fromNumpyArray(zc_filtered))
        # apply on slm
        self._dm.set_shape(
            self._slm_rasterizer.reshape_map2vector(slm_raster))
