import numpy as np 
from astropy.io import fits
from tesi_slm.camera_masters import CameraMastersAnalyzer
from tesi_slm.utils.my_tools import clean_cube_images
class TiltedPsfMeasurer():
    
    FRAMES_PER_TILT = 100
    
    def __init__(self, spoc, fname_masters=None):
        self._spoc = spoc
        self._cam = spoc._cam
        self._spoc.load_masters(fname_masters)
        self._texp_master = self._spoc._texp_master
        self._master_dark = self._spoc._master_dark
        self._master_background = self._spoc._master_background
    
    def change_cmask(self, centerYX = (550, 853), RadiusInPixel = 569):
        self._spoc.change_circular_mask(centerYX, RadiusInPixel)
        
    def measure_tilted_psf(self, j, c_span, texp = 0.125, NframeXtilt = 10, init_coeff = None):
        self._texp = texp
        self.FRAMES_PER_TILT = NframeXtilt
        if(self._texp != self._texp_master):
            print('WARNING: the selected exposure time (t_exp = %g ms) is different from the '\
                  'one used to measure dark and background (t_m = %g ms)\n'\
                  'NOTE: if t_m = 0s, image reduction is not performed.'
                  %(self._texp, self._texp_master))
        self._j_noll_idx = j
        self._c_span = c_span
        j_index = j - 2
        
        if init_coeff is None:
            #first 11 zernike starting from Z2
            init_coeff = np.zeros(9)
        self._init_coeff = np.array(init_coeff)
        
        Nmodes = len(c_span)
        self._spoc.write_zernike_on_slm(init_coeff)
        coeff = init_coeff.copy()
        
        
        frame_shape = self._cam.shape()
        self._images_3d = np.zeros((Nmodes, frame_shape[0],frame_shape[1]))#,self.FRAMES_PER_TILT))
        self._cam.setExposureTime(texp)
        
        for idx, amp in enumerate(c_span):
            coeff[j_index] = amp + init_coeff[j_index]
            self._spoc.write_zernike_on_slm(coeff)
            cube_ima = self._cam.getFutureFrames(self.FRAMES_PER_TILT).toNumpyArray()
            self._images_3d[idx] = clean_cube_images(cube_ima, self._master_dark, self._master_background)
        
            
    def save_measures(self, fname):
        hdr = fits.Header()
        hdr['T_EX_MS'] = self._texp
        hdr['N_AV_FR'] = self.FRAMES_PER_TILT
        hdr['Z_J'] = self._j_noll_idx
         
        fits.writeto(fname, self._images_3d, hdr)
        
        fits.append(fname, self._c_span)
        fits.append(fname, self._init_coeff)
        
    @staticmethod    
    def load_measures(fname):
        header = fits.getheader(fname)
        hduList = fits.open(fname)
        images_3d = hduList[0].data
        c_span = hduList[1].data
        init_coeff = hduList[2].data
            
        Nframes = header['N_AV_FR']
        texp = header['T_EX_MS']
        j_noll = header['Z_J']
        return images_3d, c_span, Nframes, texp, j_noll, init_coeff
             
# class TiltedPsfReducerTRASHME():
#
#     def __init__(self, tpm_fname, cma_fname):
#         self._texp_masters, self._fNframes, \
#          self._master_dark, self._master_background = \
#          CameraMastersAnalyzer.load_camera_masters(cma_fname)
#
#         self._images_4d, self._c_span, self._Nframes, self._texp,\
#          self._j_noll, self._init_coeff = TiltedPsfMeasurer.load_measures(tpm_fname)
#
#         err_message = 'Tilted psf and camera masters must be measured with the same texp!'
#         assert self._texp_masters == self._texp, err_message
#
#     def clean_images(self):
#         tmp_clean = np.zeros(self._images_4d.shape)
#         for idx_k in range(self._images_4d.shape[0]):
#             for idx_i in range(self._Nframes):
#                 tmp_clean[idx_k, :, :, idx_i]  = self._images_4d[idx_k,:,:,idx_i] - self._master_background - self._master_dark
#         self._clean_images_4d = tmp_clean
#
#     def save_measures(self, fname):
#         hdr = fits.Header()
#         hdr['T_EX_MS'] = self._texp
#         hdr['N_AV_FR'] = self._Nframes
#         hdr['Z_J'] = self._j_noll
#
#         fits.writeto(fname, self._clean_images_4d, hdr)
#
#         fits.append(fname, self._c_span)
#         fits.append(fname, self._init_coeff)
#
#     @staticmethod    
#     def load_measures(fname):
#         header = fits.getheader(fname)
#         hduList = fits.open(fname)
#         clean_images_4d = hduList[0].data
#         c_span = hduList[1].data
#         init_coeff = hduList[2].data
#
#         Nframes = header['N_AV_FR']
#         texp = header['T_EX_MS']
#         j_noll = header['Z_J']
#         return clean_images_4d, c_span, Nframes, texp, j_noll, init_coeff