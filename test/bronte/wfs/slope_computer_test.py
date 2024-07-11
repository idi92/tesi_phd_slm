#!/usr/bin/env python

import unittest
import numpy as np
from bronte.wfs.subaperture_set import ShSubapertureSet
from astropy.io import fits as pyfits
from bronte.wfs.slope_computer import PCSlopeComputer

class PCSlopeComputerTest(unittest.TestCase):
    cogSubaps= "../etc/20140127_173900.fits"
    tcogSubaps= "../etc/20140127_173900_thre16.fits"
#    wwpaSubaps= "../etc/20140127_173900_thre16_pow15.fits"
    cognSubaps= "../etc/20140127_173900_norm.fits"
    tcognSubaps= "../etc/20140127_173900_thre16_norm.fits"
    cogFileName= "../etc/framesCoG.fits"
    tcogFileName= "../etc/framesTCoG.fits"
#    wwpaFileName= "../etc/framesWWPA.fits"
    cognFileName= "../etc/framesCoGN.fits"
    tcognFileName= "../etc/framesTCoGN.fits"
    failFileName= "../etc/framesMustFail.fits"
    cogOutput= "/tmp/differenceCoG.fits"
    tcogOutput= "/tmp/differenceTCoG.fits"
#    wwpaOutput= "/tmp/differenceWWPA.fits"
    cognOutput= "/tmp/differenceCoGN.fits"
    tcognOutput= "/tmp/differenceTCoGN.fits"


    def setUp(self):
        self._isWfsCameraUpsideDown= False



    def _genericTest(self, subapsFile, inputFile, outputFile,
                     absoluteTolerance):
        subaps = ShSubapertureSet.restore(subapsFile)
        frame = pyfits.getdata(inputFile, 0)
        slopes = pyfits.getdata(inputFile, 1)
        sc = PCSlopeComputer(subaps, self._isWfsCameraUpsideDown)
        sc.set_frame(frame)
        ds = DisplaySlopes(subaps, self._isWfsCameraUpsideDown)
        pyfits.writeto(outputFile,
            ds.reshapeSlopesVectorIn2D((sc.slopes() - slopes)), clobber=True)
        self.assertTrue(np.allclose(sc.slopes(), slopes,
                                    atol=absoluteTolerance))


    def testSlopeComputationCoG(self):
        self._genericTest(self.cogSubaps, self.cogFileName,
                          self.cogOutput, 1e-7)


    def testSlopeComputationTCoG(self):
        self._genericTest(self.tcogSubaps, self.tcogFileName,
                          self.tcogOutput, 1e-7)

#    def testSlopeComputationWWPA(self):
#        self._genericTest(self.wwpaSubaps, self.wwpaFileName,
#                          self.wwpaOutput, 1e-7)


    def testSlopeComputationCoGN(self):
        self._genericTest(self.cognSubaps, self.cognFileName,
                          self.cognOutput, 1e-2)


    def testSlopeComputationTCoGN(self):
        self._genericTest(self.tcognSubaps, self.tcognFileName,
                          self.tcognOutput, 1e-2)


    def testSlopeComputationMustFail(self):
        subaps= ShSubapertureSet.restore(self.cogSubaps)
        frame= pyfits.getdata(self.failFileName, 0)
        slopes= pyfits.getdata(self.failFileName, 1)
        sc= PCSlopeComputer(subaps, self._isWfsCameraUpsideDown)
        sc.set_frame(frame)
        self.assertFalse(np.allclose(sc.slopes(), slopes, atol=1e-7))


    def testSpotImageMomentComputations(self):
        sc= self._genericSetup()
        res= sc.spotsSecondMomentPerPupil()
        self.assertTrue((type(res) == dict))
        res= sc.spotsMomentEccentricityPerPupil()
        self.assertTrue((type(res) == dict))
        res= sc.spotsImageMomentDictionaryPerPupil()
        self.assertTrue(isinstance(res['BLUE'].values()[0],
                                   ImageMoments))


    def _genericSetup(self):
        subaps= ShSubapertureSet.restore(self.cogSubaps)
        frame= pyfits.getdata(self.cogFileName, 0)
        sc= PCSlopeComputer(subaps, self._isWfsCameraUpsideDown)
        sc.set_frame(frame)
        return sc


    def testSpotsSecondMomentCovarianceMatrixPerPupil(self):
        sc= self._genericSetup()
        res= sc.spotsSecondMomentCovarianceMatrixPerPupil()
        self.assertEqual((2, 2), res[LaserBeam.Blue].shape)



def usedToProduceReferenceFrames(lgsw_terminal):
    import os
    from argos.snapshot.snapshot import Reader


    def save(lgsw_terminal, sntag, outputfile):
        sn= Reader(sntag)
        sn.ccdFrames.restoreLGSWFrames()
        frames=sn.ccdFrames.frames
        slopes=pyfits.getdata(os.path.join(sntag,'slopes.fits'))
        hdr=pyfits.getheader(os.path.join(sntag,'info.fits'))
        slope_offset=lgsw_terminal.calibrationManager. \
                    slope_offset(hdr['HIERARCH ARGOS.LGSW.CTRL.SLOPE_OFFSET'])
        slopes_fc=pyfits.getdata(os.path.join(sntag,'slopes_frame_counters.fits'))
        frames_fc=pyfits.getdata(os.path.join(sntag,'lgsw_ccdframes.fits'), 1)
        idx=np.where(slopes_fc==frames_fc[0])[0][0]

        pyfits.append(outputfile, frames[0], hdr)
        pyfits.append(outputfile, (slopes[idx, :]+slope_offset)[0:1048])

    cogTag='/home/argos/snapshots/measures/DX/20140209/20140209_142919'
    save(lgsw_terminal, cogTag, PCSlopeComputerTest.cogFileName)
    tcogTag='/home/argos/snapshots/measures/DX/20140209/20140209_150239'
    save(lgsw_terminal, tcogTag, PCSlopeComputerTest.tcogFileName)
    wwpaTag='/home/argos/snapshots/measures/DX/20140203/20140203_014550'
    save(lgsw_terminal, wwpaTag, PCSlopeComputerTest.wwpaFileName)
    CoGNTag='/home/argos/snapshots/measures/DX/20140209/20140209_144032'
    save(lgsw_terminal, CoGNTag, PCSlopeComputerTest.cognFileName)
    TCoGNTag='/home/argos/snapshots/measures/DX/20140209/20140209_150537'
    save(lgsw_terminal, TCoGNTag, PCSlopeComputerTest.cognFileName)


if __name__ == "__main__":
    unittest.main()