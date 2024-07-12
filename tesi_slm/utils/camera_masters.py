import numpy as np 
from astropy.io import fits

class CameraMastersMeasurer():
    
    def __init__(self, camera):
        self._cam = camera
        #self._cam_shape = camera.shape()
    
    def acquire_images(self, Nframes = 100, texp_in_ms = 0.125):
        
        self._texp = texp_in_ms
        self._Nframes = Nframes
        self._cam.setExposureTime(self._texp)
        cube = self._cam.getFutureFrames(Nframes, 1) 
        self._cube_images = cube.toNumpyArray()
    
    def save_master(self, fname, tag = 'Dark'):
        hdr = fits.Header()
        hdr['MAST'] = tag
        hdr['T_EX_MS'] = self._texp
        hdr['N_FR'] = self._Nframes
        fits.writeto(fname,self._cube_images, hdr)
    
    @staticmethod
    def load_master(fname):
        header = fits.getheader(fname)
        hduList = fits.open(fname)
        cube_images = hduList[0].data 
        Nframes = header['N_FR']
        texp = header['T_EX_MS']
        tag = header['MAST']
        return tag, texp, Nframes, cube_images
    
    def acquire_masters_at_different_texp(self, texp_array, dirname, Nframes = 20, tag='dark' ):
        for texp in texp_array:
            
            self.acquire_images(Nframes, texp)
            fname = dirname + tag +"_texp_"+ str(texp)+ "ms.fits" 
            self.save_master(fname, tag)
            

class CameraMastersAnalyzer():
    
    def __init__(self, fname_dark, fname_bg):
        self._tag_dark, self._texp_dark, self._Nframes_darks, \
         self._cube_darks = CameraMastersMeasurer.load_master(fname_dark)
        self._tag_bg, self._texp_bgs, self._Nframes_bgs, \
         self._cube_bgs = CameraMastersMeasurer.load_master(fname_bg)
        err_message1 = 'Dark and Background must be measured with the same texp!'
        assert self._texp_dark == self._texp_bgs, err_message1
        err_message2 = 'Dark and Background must be measured with the same Nframes!'
        assert self._Nframes_bgs == self._Nframes_darks, err_message2
        self._texp = self._texp_dark
    
    def compute_masters(self):
        self._compute_master_dark()
        self._compute_master_background()
        
    def _compute_master_dark(self):
        self._master_dark = np.median(self._cube_darks, axis = 2)
        
    def _compute_master_background(self):
        #frame_shape = self._cube_bgs.shape[:2]
        #self._master_background = np.zeros(frame_shape)
        tmp_master = np.zeros(self._cube_bgs.shape)
        for idx in range(self._Nframes_bgs):
            tmp_master[:, :, idx]  = self._cube_bgs[:,:,idx] - self._master_dark
        self._master_background = np.median(tmp_master, axis = 2)
    
    def save_camera_masters(self, fname):
        hdr = fits.Header()
        #hdr['MAST'] = tag
        hdr['T_EX_MS'] = self._texp
        hdr['N_FR'] = self._Nframes_darks
        fits.writeto(fname, self._master_dark, hdr)
        fits.append(fname, self._master_background)
    
    @staticmethod
    def load_camera_masters(fname):
        header = fits.getheader(fname)
        hduList = fits.open(fname)
        master_dark = hduList[0].data 
        master_background = hduList[1].data 
        Nframes = header['N_FR']
        texp = header['T_EX_MS']
        #tag = header['MAST']
        return texp, Nframes, master_dark, master_background
    
def compute_masters_at_different_texp(texp_array, fdir_dark, fdir_bg, fdir_masters):
    
        for texp in texp_array:
            fname_dark = fdir_dark + "cmm_dark_texp_"+ str(texp)+"ms.fits"
            fname_bg = fdir_bg + "cmm_bg_texp_"+ str(texp)+"ms.fits"
            cma = CameraMastersAnalyzer(fname_dark, fname_bg)
            cma.compute_masters()
            fname_masters = fdir_masters+"cma_masters_texp"+str(texp)+"ms.fits"
            cma.save_camera_masters(fname_masters)