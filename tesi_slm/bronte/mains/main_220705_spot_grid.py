import numpy as np
from astropy.io import fits
from arte.types.scalar_bidimensional_function import ScalarBidimensionalFunction
from arte.utils.discrete_fourier_transform import BidimensionalFourierTransform
from bronte.wfs.subaperture_set import ShSubapertureSet
from bronte.wfs.slope_computer import PCSlopeComputer



def load_flat_wavefront_frames():
    fname='/Volumes/GoogleDrive-100122613177551791357/Drive condivisi/LabMCAO/frame_test/220706_prove_banco/sh_wfs_1.fits'
    frames = fits.getdata(fname)
    return frames
    
def reference_frame(frames):
    mfra0 = frames.mean(axis=2)
    bias = estimate_bias(mfra0, 150) 
    return mfra0 - bias

def estimate_bias(frame, dark_square_size):
    m1 = np.median(frame[0:dark_square_size, 0:dark_square_size])
    m2 = np.median(frame[0:dark_square_size, -dark_square_size:])
    m3 = np.median(frame[-dark_square_size:, 0:dark_square_size])
    m4 = np.median(frame[-dark_square_size:, -dark_square_size:])
    print("%g %g %g %g"% (m1,m2,m3,m4))
    return np.mean([m1,m2,m3,m4])

def estimate_grid_pitch_and_shift(flat_wavefront_frame):
    ymap, xmap = np.mgrid[-512:512,-512:512]
    sbf=ScalarBidimensionalFunction(flat_wavefront_frame, xmap=xmap, ymap=ymap)
    ftsbf = BidimensionalFourierTransform.direct(sbf)
    # TODO find coords of peak in the frame fft
    # first_peak_x_coord = [512,591]
    pitch_x = (1/ftsbf.xmap[512,591])
    pitch_y = (1/ftsbf.ymap[591,512])
    dx = np.angle(ftsbf.values)[512,591] / (2*np.pi) * pitch_x
    dy = np.angle(ftsbf.values)[591,512] / (2*np.pi) * pitch_y
    print("dx %g - dy %g" % (dx, dy))
    cx, cy = (512-dx, 512-dy)
    print('pitch %g %g' % (pitch_x, pitch_y))
    print('central pixel %g %g' % (cx, cy))
    return pitch_x, pitch_y, cx, cy
    
def estimate_bottom_left_from_pitch_and_center(pitch_x, pitch_y, cx, cy, frame_sz):
    subap_sz_x = round(pitch_x)
    subap_sz_y = round(pitch_y)
    lefts = cx - 0.5*subap_sz_x - np.arange(0, frame_sz[1], subap_sz_x)
    idx_left = np.max(np.where(lefts > 0)[0])
    left = round(lefts[idx_left])
    nsubapx = int((frame_sz[1]-left)/subap_sz_x)
    bottoms = cy - 0.5*subap_sz_y - np.arange(0, frame_sz[0], subap_sz_y)
    idx_bottoms = np.max(np.where(bottoms > 0)[0])
    bottom = round(bottoms[idx_bottoms])
    nsubapy = int((frame_sz[0]-bottom)/subap_sz_y)
    print("bl %g %g" % (bottom, left))
    print("nsubap %g %g" % (nsubapx, nsubapy))
    return bottom, left, nsubapx, nsubapy

def create_subaperture_set():
    frames = load_flat_wavefront_frames()
    frame = reference_frame(frames)
    detector_shape = frame.shape
    pitch_x, pitch_y, cx, cy = estimate_grid_pitch_and_shift(frame)
    bottom, left , n_subap_x, n_subap_y = estimate_bottom_left_from_pitch_and_center(
        pitch_x, pitch_y, cx, cy, detector_shape)
    subap_size = round(pitch_x)
    n_subap = 13 
    ID_list = np.arange(n_subap_x * n_subap_y)
    bl_init = [20, 10]
    x = np.array([left + subap_size * i for i in range(n_subap_x)])
    y = np.array([bottom + subap_size * i for i in range(n_subap_y)])
    xgrid = np.repeat(x[np.newaxis, :], n_subap_y, axis=0).flatten()
    ygrid = np.repeat(y[np.newaxis, :], n_subap_x, axis=1).flatten()
    bl_list = [np.array([xgrid, ygrid])[:,i] for i in range(len(ID_list))]
    sset = ShSubapertureSet()
    subaset = sset.createMinimalSet(
        ID_list, detector_shape, subap_size, bl_list)
    return frame, subaset 
    

def create_slope_computer():
    ima, subaset = create_subaperture_set()
    sl_pc = PCSlopeComputer(subaset)
    sl_pc.set_frame(ima)
    #imshow(sl_pc.pixelsSubapsMap(), origin='lower')
    return ima, subaset, sl_pc
