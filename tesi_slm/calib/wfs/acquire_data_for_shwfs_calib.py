import numpy as np 
from tesi_slm.utils import fits_io
#from pysilico import camera
import datetime as date
from tesi_slm.utils.frame_acquisition import create_device, acquire_frames_from_camera

def main(fname_sh, fname_cam, NrevADJ, texp_sh = 4.5, texp_cam = 2.2, Nframes = 100, type_data_sh = 'WFS TILT RAW - MOUNT ADJ-Y', type_data_ccd = 'TILTED PSF RAW - MOUNT ADJ-Y'):
    
    today = str(date.datetime.today())
    fpath = "C:\\Users\labot\\Desktop\\misure_tesi_slm\\shwfs_calibration\\240717_tilt_linearity_on_subapertures\\data\\raw_data\\"
    fname_sh = fpath + fname_sh
    fname_cam = fpath + fname_cam
    
    shwfs = create_device('localhost', 7110)
    cam = create_device('localhost', 7100)
    
    shwfs.setExposureTime(texp_sh)
    cam.setExposureTime(texp_cam)
    
    fps_sh = shwfs.getFrameRate()
    fps_cam = cam.getFrameRate()

    print("\Acquiring measurements from SHWFS ...\n")
    dataCubeSH = acquire_frames_from_camera(shwfs, texp_sh, Nframes)
    print("\Done.")
    print("\Acquiring measurements from CAMERA ...\n")
    dataCubeCAM = acquire_frames_from_camera(cam, texp_cam, Nframes)
    print("\Done.")
    
    header_dict_shwfs = {
        "DATE" : today,
        "TYP_DATA" : type_data_sh,
        "NOTES" : "TILT LINEARITY MEAS IN SUBAPS - TIPTILT STAGE ADJ ROTATION",
        "NUM_REV" : NrevADJ,
        "CAM" : 'MANTA G419',
        "DEV" : 'SHWFS',
        "TEXP_MS" : texp_sh,
        "FPS" : fps_sh,
        "ON_PUPIL" : 'MIRROR WL/20',
        "Mount" : "THORLABS KM100T SM1",
        "D_EP_MM" : 10.5,
        "WL_NM" : 633
    }
    
    
    header_dict_cam = {
        "DATE" : today,
        "TYP_DATA" : type_data_ccd,
        "NOTES" : "PSF DESPLACEMENT CHECK",
        "NUM_REV" : NrevADJ,
        "CAM" : 'GC1350M',
        "DEV" : 'IMAGER',
        "TEXP_MS" : texp_cam,
        "FPS" : fps_cam,
        "ON_PUPIL" : 'MIRROR WL/20',
        "Mount" : "THORLABS KM100T SM1",
        "D_EP_MM" : 10.5,
        "WL_NM" : 633
    }
    
    vector_dict={}
    print("\nSaving...")
    fits_io.save(fname_sh, dataCubeSH, header_dict_shwfs, vector_dict)
    fits_io.save(fname_cam, dataCubeCAM, header_dict_cam, vector_dict)
    print("\Done.")
    
def demo():
    #0.5 rotation of Y adjuster 
    fname_cam = '240717cam_AdjY_Nrev0.375_raws.fits'
    fname_sh = '240717shwfs_AdjY_Nrev0.375_raws.fits'
    Nrev = 0.5
    texp_sh = 4.5
    texp_cam = 2.2
    Nframes = 100
    type_data_sh = 'WFS TILT RAW - MOUNT ADJ-Y' 
    type_data_ccd = 'TILTED PSF RAW - MOUNT ADJ-Y'
    main(fname_sh, fname_cam, Nrev, texp_sh, texp_cam, Nframes, type_data_sh, type_data_ccd)


def measure_slm_tilt(fname_sh, fname_cam, ptv_in_wl, texp_sh = 7, texp_cam = 7, Nframes = 100, type_data_sh = 'WFS TILT RAW - SLM', type_data_ccd = 'TILTED PSF RAW - SLM'):
    
    today = str(date.datetime.today())
    fpath = "C:\\Users\labot\\Desktop\\misure_tesi_slm\\shwfs_calibration\\240717_tilt_linearity_on_subapertures\\data\\raw_data\\SLM\\"
    fname_sh = fpath + fname_sh
    fname_cam = fpath + fname_cam
    
    shwfs = create_device('localhost', 7110)
    cam = create_device('localhost', 7100)
    
    shwfs.setExposureTime(texp_sh)
    cam.setExposureTime(texp_cam)
    
    fps_sh = shwfs.getFrameRate()
    fps_cam = cam.getFrameRate()

    print("\Acquiring measurements from SHWFS ...\n")
    dataCubeSH = acquire_frames_from_camera(shwfs, texp_sh, Nframes)
    print("\Done.")
    print("\Acquiring measurements from CAMERA ...\n")
    dataCubeCAM = acquire_frames_from_camera(cam, texp_cam, Nframes)
    print("\Done.")
    
    header_dict_shwfs = {
        "DATE" : today,
        "TYP_DATA" : type_data_sh,
        "NOTES" : "TILT LINEARITY MEAS IN SUBAPS - SLM BLINK",
        "PTV_WL" : ptv_in_wl,
        "CAM" : 'MANTA G419',
        "DEV" : 'SHWFS',
        "TEXP_MS" : texp_sh,
        "FPS" : fps_sh,
        "ON_PUPIL" : 'SLM',
        "WFC":'ON',
        "D_EP_MM" : 10.5,
        "WL_NM" : 633
    }
    
    
    header_dict_cam = {
        "DATE" : today,
        "TYP_DATA" : type_data_ccd,
        "NOTES" : "PSF DESPLACEMENT CHECK",
        "PTV_WL" : ptv_in_wl,
        "CAM" : 'GC1350M',
        "DEV" : 'IMAGER',
        "TEXP_MS" : texp_cam,
        "FPS" : fps_cam,
        "ON_PUPIL" : 'SLM',
        "WFC":'ON',
        "D_EP_MM" : 10.5,
        "WL_NM" : 633
    }
    
    vector_dict={}
    print("\nSaving...")
    fits_io.save(fname_sh, dataCubeSH, header_dict_shwfs, vector_dict)
    fits_io.save(fname_cam, dataCubeCAM, header_dict_cam, vector_dict)
    print("\Done.")

def acquire_device_master_bkgs(texp_sh, texp_cam, fname_sh, fname_cam):
    
    # texp_sh = 7
    # texp_cam = 7
    Nframes = 100
    
    shwfs = create_device('localhost', 7110)
    cam = create_device('localhost', 7100)
    
    cam.setExposureTime(texp_cam)
    shwfs.setExposureTime(texp_sh)
    
    fps_cam = cam.getFrameRate()
    fps_sh = shwfs.getFrameRate()
    
    today = str(date.datetime.today())
    
    header_dict_shwfs_bkg = {
        "DATE" : today,
        "TYP_DATA" : "WFS BKG MASTER",
        "CAM" : 'MANTA G419',
        "DEV" : 'SHWFS',
        "TEXP_MS" : texp_sh,
        "FPS" : fps_sh,
    }

    header_dict_cam_bkg = {
        "DATE" : today,
        "TYP_DATA" : "CAM BKG MASTER",
        "CAM" : 'GC1350M',
        "DEV" : 'CAMERA',
        "TEXP_MS" : texp_cam,
        "FPS" : fps_cam,
    }
    
    camDataCube = acquire_frames_from_camera(cam, texp_cam, Nframes)
    shDataCube = acquire_frames_from_camera(shwfs, texp_sh, Nframes)
    
    master_bkg_cam = np.median(camDataCube, axis=2)
    master_bkg_sh = np.median(shDataCube, axis=2)
    
    # fpath = "C:\\Users\\labot\\Desktop\\misure_tesi_slm\\shwfs_calibration\\240717_tilt_linearity_on_subapertures\\camera_bkgs\\"
    # fname_cam = "240730cam_bkg_master.fits"
    # fname_sh = "240730shwfs_bkg_master.fits"
    
    fits_io.save(fname_cam, master_bkg_cam, header_dict_cam_bkg, {})
    fits_io.save(fname_sh, master_bkg_sh, header_dict_shwfs_bkg, {})
    
    