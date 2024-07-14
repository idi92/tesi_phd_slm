from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from tesi_slm.utils.fits_io import save, read_info, load


def main():
    plt.close('all')
    # example of camera parameters I want to save in the fits file
    info_date = "2024/07/14"
    deviceType = "IMAGING"
    texp = 5
    fps = 30
    frame_status = "RAW"
    pupil = "SLM"
    WFC = "ON"
    STATUS = True
    frame_shape = (20, 30)

    # example of data I want to save in the fits file
    ima = 5 * np.ones(frame_shape)
    mask = np.zeros(frame_shape)
    mask[0:4, 0:1] = 1

    cmd_vector = np.ones(6) * 7.2

    bias_cmd = np.ones(6) * 0.2

    pippo = mask.copy()
    baudo = ima.copy()
    # # check masked image
    masked_ima = np.ma.array(data=ima, mask=mask, fill_value=0)

    imacube = np.ones((2, 2, 3))
    cubemask = np.array([[[1, 0, 0],
                          [0, 1, 0]],
                         [[0, 1, 0],
                          [0, 0, 1]]])
    masked_imacube = np.ma.array(data=imacube, mask=cubemask, fill_value=0)

    ima2save = ima
    ima2save = masked_ima
    ima2save = imacube
    ima2save = masked_imacube

    # creating the dictionaries
    par_dict = {
        "DATE": info_date,
        "DEV_TYPE": deviceType,
        "TEXP_MS": texp,
        "FPS": fps,
        "DAT_STAT": frame_status,
        "ON_PUPIL": pupil,
        "WFC": WFC,
        "SYS_STAT": STATUS
    }
    data_dict = {
        "FRAME_SHAPE": frame_shape,
        "CMD": cmd_vector,
        "BIAS": bias_cmd,
        "DATA_PIPPO": pippo,
        "DATA_BAUDO": baudo
    }

    fpath = "C:\\Users\\edoar\\Desktop\\pytrash\\"
    fname = fpath + "example240714_smartsaveandloadV2.fits"

    #par_dict = {}
    #data_dict = {}

    save(fname, ima2save, par_dict, data_dict)

    return ima2save, par_dict, data_dict
