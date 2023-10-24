import numpy as np
from arte.atmo.phase_screen_generator import PhaseScreenGenerator
from arte.types.mask import CircularMask
from time import sleep
from tesi_slm.my_tools import reshape_map2vector
from astropy.io import fits

class DisplayAtmOnSlm():
    
    def __init__(self,
                 fname_norm_phase_screen_cube,
                 cam,
                 mirror,
                 cmask_obj = None):
        
        self._slmFrameShape = (1152, 1920)
        self._fname_npsc = fname_norm_phase_screen_cube
        self._psg = PhaseScreenGenerator.load_normalized_phase_screens(self._fname_npsc)
        self._cam  = cam
        self._mirror = mirror
        if cmask_obj is None:
            self._build_circular_mask()
        else:
            self._cmask_obj = cmask_obj
        
    def _build_circular_mask(self,
                            centerYX = (571, 875),
                            RadiusInPixel = 571):
        cmask = CircularMask(
            frameShape = self._slmFrameShape,
            maskRadius = RadiusInPixel,
            maskCenter = centerYX)
        self._cmask_obj = cmask
    
    
    def _rescale_to_r0_in_meters(self, r0):
        self._psg.rescale_to(r0)
        return self._psg.get_in_meters()
    
    def get_short_exp_images(self, r0 = 0.2, texp = 1, tsleep = 1, bias = None):
        
        if bias is None:
            self._bias = np.zeros((1152*1920))
        else:
            self._bias = bias
        
        self._r0 = r0
        self._texp = texp
        short_exp_screens_fullframe = self._rescale_to_r0_in_meters(r0)
        self._Nframes = short_exp_screens_fullframe.shape[0]
        short_exp_screes = short_exp_screens_fullframe[:, :1152, :]
        
        cam_shape = self._cam.shape()
        self._short_exp_imacube = np.zeros((self._Nframes, cam_shape[0], cam_shape[1]))
        
        self._mirror.set_shape(bias)
        self._cam.setExposureTime(texp)
        
        for idx in range(self._Nframes):
            
            wf2display = np.ma.array(data = short_exp_screes[idx], mask = self._cmask_obj.mask())
            cmd_vector = reshape_map2vector(wf2display) + bias
            self._mirror.set_shape(cmd_vector)
            sleep(tsleep)
            self._short_exp_imacube[idx] = self._cam.getFutureFrames(1).toNumpyArray()
            sleep(tsleep)
        
        self._mirror.set_shape(bias)
        return self._short_exp_imacube
    
    def save_short_exp_images(self, fname):
        hdr = fits.Header()
        hdr['T_EX_MS'] = self._texp
        hdr['N_AV_FR'] = self._Nframes
        hdr['R_0'] = self._r0
        fits.writeto(fname, self._short_exp_imacube, hdr)
        fits.append(fname, self._bias)
    
    @staticmethod
    def load_short_exp_images(fname):
        header  = fits.getheader(fname)
        texp = header['T_EX_MS']
        Nframes = header['N_AV_FR']
        r0 = header['R_0']
        hduList = fits.open(fname)
        seima_cube = hduList[0].data
        bias = hduList[1].data
        
        return seima_cube, r0, texp, Nframes, bias 
        

class SeeingLimitedAnalyser():
    
    def __init__(self, fname):
        self._short_exp_images, self._r0, self._texp, \
            self._Nframes, self._bias = DisplayAtmOnSlm.load_short_exp_images(fname)
    
     