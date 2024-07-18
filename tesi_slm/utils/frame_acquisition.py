from pysilico import camera

def create_device(hostname = 'localhost', port = 7100):
    """
    Create and initialize a camera device using pysilico.

    This function establishes a connection to a camera device using the pysilico library. The hostname and port
    can be set through a user-defined configuration file.

    Parameters:
    ----------
    hostname : str, optional
        The hostname or IP address of the device. Default is 'localhost'. This value can be
        overridden by specifying a different hostname in the configuration file.
    port : int, optional
        The port number on which the device is connected. Default is 7100. This value can
        be overridden by specifying a different port in the configuration file.

    Returns:
    -------
    device : pysilico.camera
        An instance of the camera device initialized using the pysilico library.

    """
    # imaging localhost, 7100
    # SHWFS localhost, 7110
    device = camera(hostname, port)
    return device

def acquire_frames_from_camera(device, texp, Nframes):
    """
    Acquire a series of frames from a camera device using pysilico.

    This function captures a specified number of frames from the camera.

    Parameters:
    ----------
    device : pysilico.camera
        An instance of the camera device created by `create_device()`.
    texp : float
        The exposure time for each frame, in milliseconds.
    Nframes : int
        The number of frames to capture.

    Returns:
    -------
    dataCube : numpy.ndarray
        A 3D numpy array containing the captured frames, where the dimensions are
        (height, width, Nframes).
    """
    device.setExposureTime(texp)
    dataCube = device.getFutureFrames(Nframes).toNumpyArray()
    
    return dataCube

# header_dict_shwfs_bkg = {
#         "DATE" : today,
#         "TYP_DATA" : "WFS BKG MASTER",
#         "CAM" : 'MANTA G419',
#         "DEV" : 'SHWFS',
#         "TEXP_MS" : texp_sh,
#         "FPS" : fps_sh,
# }
#
# header_dict_cam_bkg = {
#         "DATE" : today,
#         "TYP_DATA" : "CAM BKG MASTER",
#         "CAM" : 'GC1350M',
#         "DEV" : 'CAMERA',
#         "TEXP_MS" : texp_cam,
#         "FPS" : fps_cam,
# }