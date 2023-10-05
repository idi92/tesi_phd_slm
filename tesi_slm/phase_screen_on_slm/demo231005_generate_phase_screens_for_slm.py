import numpy as np 
from arte.atmo.phase_screen_generator import PhaseScreenGenerator


def main():
    PupilDiameterInMeters = 8 # telescope pupil diameter
    PupilRadiusInPixel  = 571 # SLM pupil mask radius
    OuterScaleInMeters = 100
    seed = 9857125
    slmFrameSize = (1152, 1920)
    ScreenSizeInPixel = slmFrameSize[1]
    ScreenSizeInMeters = 0.5*PupilDiameterInMeters/PupilRadiusInPixel*slmFrameSize[1]
    
    psg = PhaseScreenGenerator(ScreenSizeInPixel, ScreenSizeInMeters, OuterScaleInMeters, seed)
    Nps = 10
    psg.generate_normalized_phase_screens(Nps)
    fname = "C:\\Users\\labot\\Desktop\\misure_tesi_slm\\display_phase_screens\\231005norm_cube_10Nps_prova.fits"
    psg.save_normalized_phase_screens(fname)
    psg_new = PhaseScreenGenerator.load_normalized_phase_screens(fname)
    #once load the normalized phase screen you can scale it
    #psg_new.rescale_to(0.2)
    #psg_new.get_in_meters()
    
    return psg_new

def generateNps4slm(                    
                    fname,
                    NPhaseScreen = 10,
                    PupilDiameterInMeters = 8,
                    PupilRadiusInPixel = 571,
                    OuterScaleInMeters = 100,
                    seed = 9857125):
    
    slmFrameSize = (1152, 1920)
    ScreenSizeInPixel = slmFrameSize[1]
    ScreenSizeInMeters = 0.5*PupilDiameterInMeters/PupilRadiusInPixel*slmFrameSize[1]
    
    psg = PhaseScreenGenerator(ScreenSizeInPixel, ScreenSizeInMeters, OuterScaleInMeters, seed)
    Nps = NPhaseScreen
    psg.generate_normalized_phase_screens(Nps)
    psg.save_normalized_phase_screens(fname)
    return psg