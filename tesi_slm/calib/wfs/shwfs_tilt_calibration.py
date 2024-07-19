import numpy as np 
from tesi_slm.utils import fits_io
from tesi_slm.utils.my_tools import get_clean_cube_images, get_index_from_image, cut_image_around_coord
import matplotlib.pyplot as plt

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
    
    def get_baricenter_from_dataCube(self, dataCube):
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
        roi_size = 20
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
        

def main():
    
    fpath = "C:\\Users\\labot\\Desktop\\misure_tesi_slm\\shwfs_calibration\\240717_tilt_linearity_on_subapertures\\data\\KM100TSM1_AdjusterX_Tip\\"
    fname_cam_ref_raw = "240717cam_AdjX_Nrev0_reference_raws.fits"
    
    fpath_bkg = "C:\\Users\\labot\\Desktop\\misure_tesi_slm\\shwfs_calibration\\240717_tilt_linearity_on_subapertures\\camera_bkgs\\"
    fname_cam_bkg = "240717cam_bkg_master_afternoon.fits"
    
    bkg, b, c = fits_io.load(fpath_bkg+fname_cam_bkg)
    rawCube, d, c = fits_io.load(fpath+fname_cam_ref_raw)
    redCube = get_clean_cube_images(rawCube, bkg, 0)
    redCube[redCube<0] = 0
    
    plt.figure()
    plt.clf()
    plt.imshow(redCube[521:541,650:670,0])
    plt.colorbar()
    pbm = PsfBaricenterMeasurer()
    
    y_bar_in_image, x_bar_in_image = pbm.get_baricenter_from_dataCube(redCube)
    
    
    return y_bar_in_image, x_bar_in_image
    

    
    