from functools import cached_property
import numpy as np
from arte.types.mask import CircularMask
from arte.utils.zernike_decomposer import ZernikeModalDecomposer
import logging
from arte.utils.decorator import logEnterAndExit


class SlmRasterizer:

    def __init__(self):
        self._logger = logging.getLogger("SlmRasterizer")
        self._zernike_modal_decomposer = ZernikeModalDecomposer(n_modes=10)

    @cached_property
    def slm_pupil_mask(self):
        centerYX = (581, 875)
        RadiusInPixel = 571
        frameshape = (1152, 1920)
        cmask = CircularMask(
            frameShape=frameshape,
            maskRadius=RadiusInPixel,
            maskCenter=centerYX)
        return cmask

    @logEnterAndExit("Converting zernike coefficients to slm map",
                     "Zernike coefficients converted to slm map", level='debug')
    def zernike_coefficients_to_raster(self, zernike_coefficients):
        '''
        Convert a ZernikeCoefficients object to a wavefront raster
        in wf meter units.
        '''
        wfz = self._zernike_modal_decomposer.recomposeWavefrontFromModalCoefficients(
            zernike_coefficients, self.slm_pupil_mask)
                
        wfz.fill_value = 0
        return wfz

    def reshape_map2vector(self, array2d, length=2211840, method='C'):
        return np.reshape(array2d, (length,), method)

    def reshape_vector2map(self, vector, shape=[1152, 1920], method='C'):
        return np.reshape(vector, (shape[0], shape[1]), method)
