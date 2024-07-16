import numpy as np 
from astropy.io import fits
from pysilico import camera
import matplotlib.pyplot as plt

from tesi_slm.bronte.wfs.slope_computer import PCSlopeComputer
from tesi_slm.bronte.wfs.subaperture_set import ShSubapertureSet
from arte.types.mask import BaseMask, CircularMask
from arte.types.slopes import Slopes
from arte.utils.modal_decomposer import ModalDecomposer
from arte.utils.rebin import rebin

from tesi_slm.utils import fits_io
import datetime as date
    



    
def main240716_acquire_wf_measurments(fname, type_data = 'WFS RAW', texp = 4.5, Nframes = 100):
    
    
    shwfs = create_shwfs_device()
    #texp = 4.5
    shwfs.setExposureTime(texp)
    fps = shwfs.getFrameRate()
    #Nframes = 100
    dataCube = acquire_image_from_shwfs(shwfs, texp, Nframes)
    
    
    header_dict = {
        "DATE" : str(date.datetime.today()),
        "TYP_DATA" : type_data,
        "CAM" : 'MANTA G419',
        "DEV" : 'SHWFS',
        "TEXP_MS" : texp,
        "FPS" : fps,
        "ON_PUPIL" : 'MIRROR WL/20',
        "D_EP_MM" : 10.5,
        "WL_NM" : 633
    }
    
    vector_dict={}
    
    fits_io.save(fname, dataCube, header_dict, vector_dict)
    
def create_shwfs_device():
    shwfs = camera('localhost', 7110)
    return shwfs

def acquire_image_from_shwfs(shwfs, texp = 5, Nframes = 100):
    
    shwfs.setExposureTime(texp)
    ima_cube = shwfs.getFutureFrames(Nframes).toNumpyArray()
    return ima_cube

def save_shwfs_ima(fname, ima, texp):
    
    hdr = fits.Header()
    hdr['T_EX_MS'] = texp
    fits.writeto(fname, ima, hdr)

def load_ima(fname):
    header = fits.getheader(fname)
    hduList = fits.open(fname)
    texp = header['T_EX_MS']
    ima = hduList[0].data
    return ima, texp

def main(Nsub = 50, subshiftXY = [0,0]):
    
    fpath = "C:\\Users\\labot\\Desktop\\misure_tesi_slm\\shwfs_calibration\\"
    fname = fpath + "240712shwfs_pupil_red.fits"
    
    ima_wfs,texp = load_ima(fname)  
    
    pixel_size = 5.5e-6 
    lenslet_size = 144e-6
    
    wl = 633e-9
    f = 6.91e-3
    beta_max = 0.5*lenslet_size/f
    theta = wl/lenslet_size
    
    pixel_per_sub = int(144/5.5) # Num of pixel on lenslet diameter
    
    #Nsub = 52 #46 active
    frame_shape = ima_wfs.shape
    
    #bottom left coordinates
    ybll =  382
    xbll = 382
    
    bll = np.mgrid[ybll:ybll+pixel_per_sub*Nsub:pixel_per_sub,
                    xbll:xbll+pixel_per_sub*Nsub:pixel_per_sub].reshape((2, Nsub*Nsub)).T
                    
    subaps = ShSubapertureSet.createMinimalSet(
        np.arange(Nsub*Nsub), frame_shape, pixel_per_sub, bll)
    
    sc = PCSlopeComputer(subaps)
    sc.set_frame(ima_wfs)
    
    # plot spots and subap grid
    plt.figure()
    plt.clf()
    plt.imshow(sc.subapertures_map()*1000+sc.frame())
    plt.colorbar()
    
    # shift subapgrid if needed to center the spots
    subaps.shiftSubap(subaps.keys(), subshiftXY)
    plt.figure()
    plt.clf()
    plt.title("Subapetrure shift DeltaXY: %d , %d" %(subshiftXY[0],subshiftXY[1]))
    plt.imshow(sc.subapertures_map()*1000 + sc.frame())
    plt.colorbar()
    
    plt.figure()
    plt.clf()
    plt.imshow(sc.subapertures_flux_map())
    plt.colorbar()
    
    sub_flux = sc.subapertures_flux_map().flatten()
    # plot an Histogram to set a threshold
    plt.figure()
    plt.clf()
    plt.hist(sub_flux, bins=range(0, int(sub_flux.max()), int(sub_flux.max()*0.02)), fc='k', ec='k')
    plt.xlabel('Sum Flux [ADU]')
    plt.ylabel('N')
    sc.remove_low_flux_subaps(threshold=50000)
    
    Nsub_tot = sc.total_number_of_subapertures()
    
    plt.figure()
    plt.clf()
    plt.title("After SubAperture removal: %d tot sub" % Nsub_tot)
    plt.imshow(sc.subapertures_map()*1000+sc.frame())
    plt.colorbar()
    
    plt.figure()
    plt.clf()
    plt.title("After SubAperture removal: %d tot sub" % Nsub_tot)
    plt.imshow(sc.subapertures_flux_map())
    plt.colorbar()