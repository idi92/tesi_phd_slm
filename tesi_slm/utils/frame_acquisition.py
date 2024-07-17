from pysilico import camera

def create_device(hostname = 'localhost', port = 7100):
    # imaging localhost, 7100
    # SHWFS localhost, 7110
    device = camera(hostname, port)
    return device

def acquire_frames_from_camera(device, texp, Nframes):
    
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