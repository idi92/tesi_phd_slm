from pysilico import camera
from plico_dm import deformableMirror
from functools import cached_property

from tesi_slm.bronte import package_data
from tesi_slm.bronte.wfs.rtc import ScaoRealTimeComputer
from tesi_slm.bronte.wfs.slm_rasterizer import SlmRasterizer
from tesi_slm.bronte.wfs.slope_computer import PCSlopeComputer
from tesi_slm.bronte.wfs.subaperture_set import ShSubapertureSet

from arte.utils.modal_decomposer import ModalDecomposer
from arte.atmo.phase_screen_generator import PhaseScreenGenerator
from tesi_slm.bronte.wfs.temporal_controller import PureIntegrator


class BronteFactory():
    SUBAPS_TAG = '240806_120800'  # '240802_122800'
    PHASE_SCREEN_TAG = '240806_124700'

    def __init__(self):
        self._set_up_basic_logging()
        self._create_phase_screen_generator()
        self._subaps = ShSubapertureSet.restore(
            package_data.subaperture_set_folder() / (self.SUBAPS_TAG+'.fits'))
        self._sc = PCSlopeComputer(self._subaps)

    def _set_up_basic_logging(self):
        import importlib
        import logging
        importlib.reload(logging)
        FORMAT = '%(asctime)s:%(levelname)s:%(name)s  %(message)s'
        logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    def _create_phase_screen_generator(self):
        r0 = 0.3
        self._psg = PhaseScreenGenerator.load_normalized_phase_screens(
            package_data.phase_screen_folder() / (self.PHASE_SCREEN_TAG+'.fits'))
        self._psg.rescale_to(r0)

    @cached_property
    def sh_camera(self):
        return camera('193.206.155.69', 7110)

    @cached_property
    def psf_camera(self):
        return camera('193.206.155.69', 7100)

    @cached_property
    def deformable_mirror(self):
        return deformableMirror('193.206.155.69', 7010)

    @cached_property
    def subapertures_set(self):
        return self._subaps

    @cached_property
    def slope_computer(self):
        return self._sc

    @cached_property
    def slm_rasterizer(self):
        return SlmRasterizer()

    @cached_property
    def modal_decomposer(self):
        return ModalDecomposer(50)

    @cached_property
    def pure_integrator_controller(self):
        return PureIntegrator()

    @cached_property
    def rtc(self):
        return ScaoRealTimeComputer(self.sh_camera, self.slope_computer, self.deformable_mirror, self.modal_decomposer, self.pure_integrator_controller, self.slm_rasterizer)

    @cached_property
    def phase_screen_generator(self):
        return self._psg
