from matplotlib import pyplot as plt
import numpy as np

from tesi_slm.bronte.wfs.slope_computer import PCSlopeComputer
from tesi_slm.bronte.wfs.subaperture_set import ShSubapertureSet
from arte.types.mask import BaseMask, CircularMask
from arte.types.slopes import Slopes
from arte.utils.modal_decomposer import ModalDecomposer
from arte.utils.rebin import rebin


def main(wfs_camera_frame):
    # wfs_camera_frame=cam.getFutureFrames(1).toNumpyArray()
    bll = np.mgrid[0:0+26*70:26, 0:0+26*70:26].reshape((2, 70*70)).T
    subaps = ShSubapertureSet.createMinimalSet(
        np.arange(4900), (2048, 2048), 26, bll)
    sc = PCSlopeComputer(subaps)
    sc.set_frame(wfs_camera_frame)

    # plot spots and subap grid
    plt.clf()
    plt.imshow(sc.subapertures_map()*1000+sc.frame())
    plt.colorbar()

    # shift subapgrid if needed to center the spots
    subaps.shiftSubap(subaps.keys(), [0, 8])

    # remove low flux subaps
    sc.remove_low_flux_subaps()

    # remove edge subaps that are low flux but not automatically deleted
    sc.remove_low_flux_subaps(threshold=60000)

    # compute Zernikes
    md = ModalDecomposer(100)
    mask = CircularMask((45, 45))

    # ugly, need rebinned subapertures map
    dd = rebin(sc.subapertures_weights_map()[
               460:460+45*26, 432:432+45*26], (45, 45))
    maska = (1-dd).astype(bool)

    # create Slopes object
    sl = Slopes(sc.slopes()[:, 0], sc.slopes()[:, 1], maska)

    # use modal decomposer
    zc = md.measureZernikeCoefficientsFromSlopes(sl, mask, BaseMask(maska))

    plt.plot(np.arange(100)+2, zc.toNumpyArray())
