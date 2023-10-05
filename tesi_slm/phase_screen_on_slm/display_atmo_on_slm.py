import numpy as np
from arte.atmo.phase_screen_generator import PhaseScreenGenerator
from arte.utils.zernike_generator import ZernikeGenerator
from arte.types.mask import CircularMask

class DisplayAtmOnSlm():
    
    def __init__(self,
                 fname_norm_phase_screen_cube,
                 cmask_obj = None):
        
        self._slmFrameShape = (1152, 1920)
        self._fname_npsc = fname_norm_phase_screen_cube
        self._psg = PhaseScreenGenerator.load_normalized_phase_screens(self._fname_npsc)
        if cmask_obj is None:
            self._build_circular_mask()
        
    def _build_circular_mask(self,
                            centerYX = (571, 875),
                            RadiusInPixel = 571):
        cmask = CircularMask(
            frameShape = self._slmFrameShape,
            maskRadius = RadiusInPixel,
            maskCenter = centerYX)
        self._cmask_obj = cmask
    
    
    