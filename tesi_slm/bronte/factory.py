from pysilico import camera
from plico_dm import deformableMirror
from functools import cached_property

from tesi_slm.bronte import package_data
from tesi_slm.bronte.wfs.rtc import ScaoRealTimeComputer
from tesi_slm.bronte.wfs.slm_rasterizer import SlmRasterizer
from tesi_slm.bronte.wfs.slope_computer import PCSlopeComputer
from tesi_slm.bronte.wfs.subaperture_set import ShSubapertureSet

from arte.utils.modal_decomposer import ModalDecomposer

from tesi_slm.bronte.wfs.temporal_controller import PureIntegrator


class BronteFactory():
    SUBAPS_TAG = '240805_191000'  # '240802_122800'

    def __init__(self):
        self._set_up_basic_logging()
        self._subaps = ShSubapertureSet.restore(
            package_data.subaperture_set_folder() / (self.SUBAPS_TAG+'.fits'))
        self._sc = PCSlopeComputer(self._subaps)

    def _set_up_basic_logging(self):
        import importlib
        import logging
        importlib.reload(logging)
        logging.basicConfig(level=logging.DEBUG)

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
