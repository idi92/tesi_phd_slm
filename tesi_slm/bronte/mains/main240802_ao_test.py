import numpy as np
from tesi_slm.calib.wfs import shwfs_tilt_calibration
import matplotlib.pyplot as plt
from arte.types.zernike_coefficients import ZernikeCoefficients
import time
import logging


def define_subap_set(shwfs, slm, flux_threshold=100000):
    '''
    used to save subapertures 240802_122800
    '''
    slm.set_shape(np.zeros(1152*1920))
    wf_ref = shwfs.getFutureFrames(1, 20).toNumpyArray()
    sgi = shwfs_tilt_calibration.main(wf_ref, flux_threshold=flux_threshold)
    return sgi


class TestAoLoop:
    SLM_RESPONSE_TIME = 0.005

    def __init__(self, factory):
        self._factory = factory
        self._logger = logging.getLogger("TestAoLoop")

    def loop(self, how_many=10):
        for i in range(how_many):
            self._logger.info("loop %d/%d" % (i+1, how_many))
            self.step()
            self.display()

    def step(self):
        self._factory.rtc.step()
        time.sleep(self.SLM_RESPONSE_TIME)

    def display(self):

        psf_ima = self._factory.psf_camera.getFutureFrames(1, 1).toNumpyArray()
        plt.figure(1)
        plt.clf()
        plt.imshow(psf_ima)
        plt.colorbar()

        sh_ima = self._factory.sh_camera.getFutureFrames(1, 1).toNumpyArray()
        plt.figure(2)
        plt.clf()
        plt.imshow(self._factory.slope_computer.subapertures_map()*1000+sh_ima)
        plt.colorbar()

        plt.figure(3)
        self.show_slopes_x_maps()
        plt.figure(4)
        self.show_slopes_y_maps()

        plt.figure(5)
        plt.clf()
        plt.plot(self._factory.slope_computer.slopes()[:, 0])
        plt.plot(self._factory.slope_computer.slopes()[:, 1])

        plt.figure(6)
        plt.clf()
        plt.imshow(self._factory.slm_rasterizer.reshape_vector2map(
            self._factory.deformable_mirror.get_shape()))
        plt.colorbar()

        plt.figure(7)
        zc = ZernikeCoefficients.fromNumpyArray(
            self._factory.pure_integrator_controller.command())
        plt.plot(zc.zernikeIndexes(), zc.toNumpyArray(), '.-')
        plt.grid(True)
        plt.ylabel('integrated modal coefficient')
        plt.xlim(2, 20)

        plt.figure(8)
        zc = self._factory.rtc._compute_zernike_coefficients()
        plt.plot(zc.zernikeIndexes(), zc.toNumpyArray(), '.-')
        plt.grid(True)
        plt.ylabel('delta modal coefficient')
        plt.xlim(2, 20)

    def show_slopes_x_maps(self):
        sc = self._factory.slope_computer
        plt.clf()
        plt.title("Slope X")
        plt.imshow(sc.slopes_x_map())
        plt.colorbar()

    def show_slopes_y_maps(self):
        sc = self._factory.slope_computer
        plt.clf()
        plt.title("Slope Y")
        plt.imshow(sc.slopes_y_map())
        plt.colorbar()
