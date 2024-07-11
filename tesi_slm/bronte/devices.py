from pysilico import camera
from plico_dm import deformableMirror
from plico_interferometer import interferometer 
from plico.utils.decorator import cacheResult

class BronteFactory():
    
    def __init__(self):
        pass
 
    @property
    @cacheResult   
    def sh_camera(self):
        return camera('193.206.155.132', 7100)

    @property
    @cacheResult   
    def psf_camera(self):
        return camera('193.206.155.132', 7110)
    
    @property
    @cacheResult   
    def deformable_mirror(self):
        return deformableMirror('193.206.155.92', 7000)

    @property
    @cacheResult   
    def wyko(self):
        return interferometer('193.206.155.29', 7300)
        