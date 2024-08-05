from functools import cached_property
import numpy as np
from arte.types.mask import CircularMask
from arte.utils.zernike_generator import ZernikeGenerator


class SlmRasterizer:

    def __init__(self):
        self._zernike_generator = ZernikeGenerator(self.slm_pupil_mask)

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

    def zernike_coefficients_to_raster(self, zernike_coefficients):
        '''
        Convert a ZernikeCoefficients object to a wavefront raster
        in wf meter units.
        '''
        wfz = np.ma.array(
            data=np.zeros((1152, 1920)), mask=self.slm_pupil_mask.mask(), fill_value=0)

        for y in zernike_coefficients.zernikeIndexes():
            wfz += zernike_coefficients.getZ([y]) * \
                self._zernike_generator.getZernike(y)
                
        wfz.fill_value = 0
        return wfz

    def reshape_map2vector(self, array2d, length=2211840, method='C'):
        return np.reshape(array2d, (length,), method)

    def reshape_vector2map(self, vector, shape=[1152, 1920], method='C'):
        return np.reshape(vector, (shape[0], shape[1]), method)
