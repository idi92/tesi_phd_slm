import numpy as np
from tesi_slm.calib.wfs import shwfs_tilt_calibration
import matplotlib.pyplot as plt
from arte.types.zernike_coefficients import ZernikeCoefficients
import time
import logging


def define_subap_set(shwfs, slm, corner_xy=(0, 0), nsubaps=50, flux_threshold=100000):
    '''
    used to save subapertures 240802_122800
    '''
    slm.set_shape(np.zeros(1152*1920))
    wf_ref = shwfs.getFutureFrames(1, 20).toNumpyArray()
    sgi = shwfs_tilt_calibration.main(
        wf_ref, corner_xy=corner_xy, nsubaps=nsubaps, flux_threshold=flux_threshold)
    return sgi


class TestAoLoop:
    SLM_RESPONSE_TIME = 0.005

    def __init__(self, factory):
        self._factory = factory
        self._logger = logging.getLogger("TestAoLoop")
        self._display_in_loop = False
        self._t = 0
        self._wavefront_disturb = None
        self.setup_disturb()
        self._long_exp = 0
        plt.ion()

    def enable_display_in_loop(self, true_or_false):
        self._display_in_loop = true_or_false

    def setup_disturb(self):
        ps = self._factory.phase_screen_generator.get_in_meters()[0, 0:1152, :]
        self.load_wavefront_disturb(ps)


    def loop(self, how_many=10):
        for i in range(how_many):
            self._logger.info("loop %d/%d" % (i+1, how_many))
            self.step()
            self.integrate_long_exposure()
            if self._display_in_loop:
                self.display()
        self.display()

    def step(self):
        self._t += 1
        self.set_wavefront_disturb(self._t)
        self._factory.rtc.step()
        time.sleep(self.SLM_RESPONSE_TIME)

    def set_wavefront_disturb(self, temporal_step):
        if self._wavefront_disturb is None:
            return
        wind_speed = 4  # in phase screen/step
        roll_by = temporal_step * wind_speed
        self._factory.rtc.set_wavefront_disturb(
            np.roll(self._wavefront_disturb, (roll_by, 0))
        )

    def load_wavefront_disturb(self, wavefront_disturb):
        self._wavefront_disturb = wavefront_disturb

    def reset_wavefront_disturb(self):
        self._factory.rtc.reset_wavefront_disturb()
        self._wavefront_disturb = None

    def integrate_long_exposure(self):
        self._short_exp = self._factory.psf_camera.getFutureFrames(
            1, 1).toNumpyArray()
        self._long_exp += self._short_exp.astype(float)

    def reset_long_exposure(self):
        self._long_exp = 0

    def display(self):
        plt.figure(1)
        plt.clf()
        plt.imshow(self._short_exp[340:440, 730:830])
        plt.colorbar()
        plt.show(block=False)
        plt.pause(0.2)

        sh_ima = self._factory.sh_camera.getFutureFrames(1, 1).toNumpyArray()
        plt.figure(2)
        plt.clf()
        plt.imshow(self._factory.slope_computer.subapertures_map()*1000+sh_ima)
        plt.colorbar()
        plt.show(block=False)
        plt.pause(0.2)

        plt.figure(3)
        self.show_slopes_x_maps()
        plt.figure(4)
        self.show_slopes_y_maps()

        plt.figure(5)
        plt.clf()
        plt.plot(self._factory.slope_computer.slopes()[:, 0])
        plt.plot(self._factory.slope_computer.slopes()[:, 1])
        plt.show(block=False)
        plt.pause(0.2)

        plt.figure(6)
        plt.clf()
        plt.imshow(self._factory.slm_rasterizer.reshape_vector2map(
            self._factory.deformable_mirror.get_shape()))
        plt.colorbar()
        plt.show(block=False)
        plt.pause(0.2)

        plt.figure(7)
        plt.clf()
        zc = ZernikeCoefficients.fromNumpyArray(
            self._factory.pure_integrator_controller.command())
        plt.plot(zc.zernikeIndexes(), zc.toNumpyArray(), '.-')
        plt.grid(True)
        plt.ylabel('integrated modal coefficient')
        plt.xlim(2, 20)
        plt.show(block=False)
        plt.pause(0.2)

        plt.figure(8)
        plt.clf()
        zc = self._factory.rtc._compute_zernike_coefficients()
        plt.plot(zc.zernikeIndexes(), zc.toNumpyArray(), '.-')
        plt.grid(True)
        plt.ylabel('delta modal coefficient')
        plt.xlim(2, 20)
        plt.show(block=False)
        plt.pause(0.2)


    def show_slopes_x_maps(self):
        sc = self._factory.slope_computer
        plt.clf()
        plt.title("Slope X")
        plt.imshow(sc.slopes_x_map())
        plt.colorbar()
        plt.show(block=False)
        plt.pause(0.2)

    def show_slopes_y_maps(self):
        sc = self._factory.slope_computer
        plt.clf()
        plt.title("Slope Y")
        plt.imshow(sc.slopes_y_map())
        plt.colorbar()
        plt.show(block=False)
        plt.pause(0.2)
