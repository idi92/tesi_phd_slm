import numpy as np 
from tesi_slm.utils import fits_io
from tesi_slm.utils.my_tools import clean_cube_images, get_index_from_image, cut_image_around_coord
import matplotlib.pyplot as plt
#import datetime as date

class PsfBaricenterMeasurer():
    
    def __init__(self):
        
        pass
        
    def get_roi(self, image, y_roi_center, x_roi_center, roi_size):
        
        half_side = int(roi_size * 0.5)
        image_roi = cut_image_around_coord(image, y_roi_center, x_roi_center, half_side) 
        return image_roi
    
    def get_baricenter(self, image):
        
        frame_size = image.shape
        yi, xi = np.mgrid[:frame_size[0], :frame_size[1]]
        Ii = image.flatten()
        Itot = Ii.sum()
        x_bar = (Ii*xi.flatten()).sum()/Itot
        y_bar = (Ii*yi.flatten()).sum()/Itot
        return y_bar, x_bar
    
def get_baricenter_from_dataCube(dataCube, roi_size = 20):
    """
    This works only if the dataCube is related to a PSF in a fixed position
    if the dataCube contains shifted psf the selected roi becames wrong
    if the psf is fixed the roi must be fixed as well for all the repeated frames
    to have consisyency in the baricenter measurements
    """
    dataCube[dataCube<0] = 0
    Nframes = dataCube.shape[-1]
    
    mean_image = dataCube.mean(axis = 2)
    ymean, xmean = get_index_from_image(mean_image, mean_image.max())
    #roi_size = 20
    pbm = PsfBaricenterMeasurer()
    
    y_bar = np.zeros(Nframes)
    x_bar = np.zeros(Nframes)
    
    for idx in range(Nframes):
        
        image = dataCube[:,:,idx]
        roi = pbm.get_roi(image, ymean, xmean, roi_size)
        y_bar_in_roi, x_bar_in_roi = pbm.get_baricenter(roi)
    
        x_bar[idx] = xmean - int(0.5*roi_size) + x_bar_in_roi
        y_bar[idx] = ymean - int(0.5*roi_size) + y_bar_in_roi
        
    return y_bar, x_bar

def get_baricenter_from_frame(frame, roi_size = 20):
    
    frame[frame<0] = 0
    ymean, xmean = get_index_from_image(frame, frame.max())
  
    pbm = PsfBaricenterMeasurer()
    
    roi = pbm.get_roi(frame, ymean, xmean, roi_size)
    y_bar_in_roi, x_bar_in_roi = pbm.get_baricenter(roi)

    x_bar = xmean - int(0.5*roi_size) + x_bar_in_roi
    y_bar = ymean - int(0.5*roi_size) + y_bar_in_roi
    
    return y_bar, x_bar
    

def main_cam_data_red():
    
    fpath = "C:\\Users\\labot\\Desktop\\misure_tesi_slm\\shwfs_calibration\\240717_tilt_linearity_on_subapertures\\data\\KM100TSM1_AdjusterX_Tip\\"
    #fpath = "C:\\Users\\labot\\Desktop\\misure_tesi_slm\\shwfs_calibration\\240717_tilt_linearity_on_subapertures\\data\\KM100TSM1_AdjusterY_Tilt\\"
    
    fpath_bkg = "C:\\Users\\labot\\Desktop\\misure_tesi_slm\\shwfs_calibration\\240717_tilt_linearity_on_subapertures\\camera_bkgs\\"
    fname_cam_bkg = "240717cam_bkg_master_afternoon.fits"
    
    bkg, b, c = fits_io.load(fpath_bkg+fname_cam_bkg)
    
    fname_cam_list = [
    
         "240717cam_AdjX_Nrev0_reference_raws.fits",
         "240717cam_AdjX_Nrev0.125_raws.fits",
         "240717cam_AdjX_Nrev0.250_raws.fits",
         #"240717cam_AdjX_Nrev0.375_raws.fits",
         "240717cam_AdjX_Nrev0.500_raws.fits",
         "240717cam_AdjX_Nrev0.625_raws.fits",
         "240717cam_AdjX_Nrev0.750_raws.fits"]
    Nrev = np.array([0, 0.125, 0.25, 0.5, 0.625, 0.75])
    
    # fname_cam_list = [
    #
    #     "240717cam_AdjY_Nrev0.000_raws.fits",
    #     "240717cam_AdjY_Nrev0.125_raws.fits",
    #     "240717cam_AdjY_Nrev0.250_raws.fits",
    #     "240717cam_AdjY_Nrev0.375_raws.fits"
    # ]
    # Nrev = np.array([0, 0.125, 0.25, 0.375])

    frame_shape = bkg.shape
    RedFrameCube = np.zeros((*frame_shape,len(fname_cam_list)))
    x_bar = np.zeros(len(fname_cam_list))
    y_bar = np.zeros(len(fname_cam_list))
    
    for idx, fname in enumerate(fname_cam_list):
    
        rawCube, d, c = fits_io.load(fpath+fname)
        redFrame = clean_cube_images(rawCube, bkg, 0)
        redFrame[redFrame<0] = 0
        y_bar[idx], x_bar[idx] = get_baricenter_from_frame(redFrame, roi_size=20)
        RedFrameCube[:,:,idx] = redFrame
    
    fpath_red ="C:\\Users\\labot\\Desktop\\misure_tesi_slm\\shwfs_calibration\\240717_tilt_linearity_on_subapertures\\data\\" 
    fname_red_data = fpath_red +  "red_data\\240717tilted_psf_on_camera_adjX.fits"
    #fname_red_data = fpath_red +  "red_data\\240717tilted_psf_on_camera_adjY.fits"
    
    today = d["DATE"]
    texp_cam = d["TEXP_MS"]
    fps_cam = d["FPS"]
    
    header_dict = {
        "DATE" : today,
        "TYP_DATA" : "TILTED PSF RED CUBE - ADJ-X",
        #"TYP_DATA" : "TILTED PSF RED CUBE - ADJ-Y",
        "CAM" : 'GC1350M',
        "DEV" : 'CAMERA',
        "TEXP_MS" : texp_cam,
        "FPS" : fps_cam,
    }
    vector_dict = {
        "Y_BAR" : y_bar,
        "X_BAR" : x_bar,
        "NREV" : Nrev,
        }
    
    fits_io.save(fname_red_data, RedFrameCube, header_dict, vector_dict)
    
    return RedFrameCube, y_bar, x_bar, Nrev
    
def main_shwfs_data_red():
    
    fpath = "C:\\Users\\labot\\Desktop\\misure_tesi_slm\\shwfs_calibration\\240717_tilt_linearity_on_subapertures\\data\\KM100TSM1_AdjusterX_Tip\\"
    #fpath = "C:\\Users\\labot\\Desktop\\misure_tesi_slm\\shwfs_calibration\\240717_tilt_linearity_on_subapertures\\data\\KM100TSM1_AdjusterY_Tilt\\"
    
    fpath_bkg = "C:\\Users\\labot\\Desktop\\misure_tesi_slm\\shwfs_calibration\\240717_tilt_linearity_on_subapertures\\camera_bkgs\\"
    fname_shwfs_bkg = "240717shwfs_bkg_master_afternoon.fits"
    
    bkg, b, c = fits_io.load(fpath_bkg+fname_shwfs_bkg)
    
    fname_shwfs_list = [
        "240717shwfs_AdjX_Nrev0_reference_raws.fits",
        "240717shwfs_AdjX_Nrev0.125_raws.fits",
        "240717shwfs_AdjX_Nrev0.250_raws.fits",
        #"240717shwfs_AdjX_Nrev0.375_raws.fits",
        "240717shwfs_AdjX_Nrev0.500_raws.fits",
        "240717shwfs_AdjX_Nrev0.625_raws.fits",
        "240717shwfs_AdjX_Nrev0.750_raws.fits"
        ]
    
    Nrev = np.array([0, 0.125, 0.25, 0.5, 0.625, 0.75])
    
    # fname_shwfs_list = [
    #
    #     "240717shwfs_AdjY_Nrev0.000_raws.fits",
    #     "240717shwfs_AdjY_Nrev0.125_raws.fits",
    #     "240717shwfs_AdjY_Nrev0.250_raws.fits",
    #     "240717shwfs_AdjY_Nrev0.375_raws.fits"
    # ]
    # Nrev = np.array([0, 0.125, 0.25, 0.375])
    
    frame_shape = bkg.shape
    RedFrameCube = np.zeros((*frame_shape,len(fname_shwfs_list)))
  
    for idx, fname in enumerate(fname_shwfs_list):
        
        rawCube, d, c = fits_io.load(fpath+fname)
        redFrame = clean_cube_images(rawCube, bkg, 0)
        redFrame[redFrame<0] = 0
        RedFrameCube[:,:,idx] = redFrame
    
    fpath_red ="C:\\Users\\labot\\Desktop\\misure_tesi_slm\\shwfs_calibration\\240717_tilt_linearity_on_subapertures\\data\\" 
    fname_red_data = fpath_red + "red_data\\240717shwfs_tilt_calib_adjX.fits"
    #fname_red_data = fpath_red + "red_data\\240717shwfs_tilt_calib_adjY.fits"
    
    today = d["DATE"]
    texp_shwfs = d["TEXP_MS"]
    fps_shwfs = d["FPS"]
    
    header_dict = {
        "DATE" : today,
        "TYP_DATA" : "SHWFS RED CUBE - ADJ-X",
        #"TYP_DATA" : "SHWFS RED CUBE - ADJ-Y",
        "CAM" : 'MANTA G419',
        "DEV" : 'SHWFS',
        "TEXP_MS" : texp_shwfs,
        "FPS" : fps_shwfs,
    }
    vector_dict = {
        "NREV" : Nrev
        }
    
    fits_io.save(fname_red_data, RedFrameCube, header_dict, vector_dict)
    
    return RedFrameCube, Nrev
        
    