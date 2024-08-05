
import numbers
# import copy
import os.path
import numpy as np
from astropy.io import fits


class ShSubaperture():
    '''
    Represents a SH subaperture.
    A subaperture is a block of NxN pixels which will result in two slopes (X
    and Y) being calculated.
    This class does force a square geometry (size=N).

    A subaperture knows:
       - its unique identifier
       - its position on the CCD pixel grid
       - the pixel weights to apply to each of its pixels
       - the adaptive and fixed centroid calculation thresholds
       - its own linearizing coefficient

    In addition to the above parameters, a subaperture can also return:
       - the pointer array to fill the SubapPixelPtr vector
       - the pointer array to fill the SubapPixelWeight vector

    Internal variables:

    ID:             unique subaperture identifier
    idx_pixel_list: row-major 1D array [size*size].
    pixel_weight:   row-major 1D array [size*size], default=1.
    fixThreshold:   threshold constant (scalar, default=0.)
    maxGain:        threshold gain. Set to 1 to have a threshold
                    proportional to subaperture's flux
                    (scalar, default=0.)
    powerCoeff:     power coefficient n. Can be 1 or 1.5.
                    slopes = c Sum_p (I_p)^n
                    (default=1 i.e. pure photocenter)
    linearCoeff:    linearization cofficient gamma (default=1)

    All vectors/array are numpy arrays. No lists.

    '''

    def __init__(self,
                 ID,
                 idxPixelList,
                 detector_shape,
                 subaperture_size,
                 pixelWeight=None,
                 fixThreshold=0.,
                 maxGain=0.,
                 powerCoeff=1.,
                 linearCoeff=1.,
                 # pupilCoords=np.zeros(2),
                 normalizationCoeff=0.
                 ):
        '''
        Initializes a subaperture. Required parameters:

        - ID:           unique identifier. Any hashable value is accepted.
        - idxPixelList: 1d pixel CCD index array (or list) [size*size]
        - detector_shape: shape of the detector (y,x)
        - subaperture_size: assuming square subaperture, in pixel

        Optional parameters:

        - pixelWeight:    1d pixel weight array. Default to all 1 if not given
        - fixThreshold:   fixed threshold for centroid computation.
                          Defaults to 0
        - adaptThreshold: adaptive threshold for centroid computation. Defaults
                          to 0
        - linearCoeff:    linearizing coefficient. Defaults to 1
        - normalizationCoeff:  normalization to total(1) or subaperture (0)
                            Defaults to 0
        '''

        self.ccdx = detector_shape[1]
        self.ccdy = detector_shape[0]
        self._size = subaperture_size

        self._ID = ID

        # Convert input array/lists to ndarray
        # idxPixelList = np.asarray(idxPixelList).flatten('F')  # COLUMN MAJOR
        # if pixelWeight is not None:
        # pixelWeight = np.asarray(pixelWeight).flatten('F')  # COLUMN MAJOR

        # Parameters validation
        if ID == None:
            raise ValueError('ID cannot be undefined')

        assert (self._size * self._size,) == idxPixelList.shape
        if pixelWeight is not None:
            assert (self._size * self._size,) == pixelWeight.shape

        self._idx_cpixel = idxPixelList

        if pixelWeight is None:
            self.setPixelWeight(np.ones(self._size * self._size))
        else:
            self.setPixelWeight(pixelWeight)

        self.setFixThreshold(fixThreshold)
        self.setMaxGain(maxGain)
        self.setPowerCoeff(powerCoeff)
        self.setLinearCoeff(linearCoeff)
        # self.setPupilCoords(pupilCoords)
        self.setNormalizationCoeff(normalizationCoeff)

    def __eq__(self, other):
        if self.ID() != other.ID() or \
           (self.pixelList() != other.pixelList()).any() or \
           (self.pixelWeight() != other.pixelWeight()).any() or \
           self.fixThreshold() != other.fixThreshold() or \
           self.maxGain() != other.maxGain() or \
           self.linearCoeff() != other.linearCoeff() or \
           self.normalizationCoeff() != other.normalizationCoeff():
            # or (self.pupilCoords() != other.pupilCoords()).any():
            return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def ID(self):
        '''
        Return the subap ID
        '''
        return self._ID

    def size(self):
        '''
        Return linear dimension of subaperture [pixels]
        '''
        return self._size

    def pixelList(self):
        '''
        Return list of pixel indexes in the BCU pnCCD standard pixel numbering
        as an ndarray (size**2,) column-major order
        '''
        return self._idx_cpixel

    def setPixelList(self, idPixelList):
        self._idx_cpixel = idPixelList

    def pixelWeight(self):
        '''
        Return pixel weight as an ndarray (size**2,), column-major order
        '''
        return self._pixelWeight

    def pixelMask(self):
        '''
        Return pixel mask (pixels value from 1 to size**2 in subap, 0 outside)
        as a ndarray(ccdy,ccdx)
        '''
        mask = np.zeros(self.ccdx * self.ccdy)
        mask[self._idx_cpixel] = np.asarray(range(len(self._idx_cpixel))) + 1
        return np.reshape(mask, (self.ccdx, self.ccdy))

    def weightMask(self):
        '''
        Return weight mask as a ndarray(ccdy,ccdx)
        '''
        wmask = np.zeros(self.ccdx * self.ccdy)
        wmask[self._idx_cpixel] = self._pixelWeight
        return np.reshape(wmask, (self.ccdx, self.ccdy))

    # def minColumnMajorFlipIdx(self):
    #     colNumber = max(self._idx_cpixel) / self.ccdx
    #     rowNumber = max(self._idx_cpixel) % self.ccdx
    #     return (self.ccdx - 1 - colNumber) * self.ccdx + rowNumber - self._size + 1
    #
    # def maxColumnMajorFlipIdx(self):
    #     colNumber = min(self._idx_cpixel) / self.ccdx
    #     rowNumber = min(self._idx_cpixel) % self.ccdx
    #     return (self.ccdx - 1 - colNumber) * self.ccdx + rowNumber + self._size - 1

    def maxGain(self):
        '''
        Return the adaptive threshold for this subaperture
        '''
        return self._maxGain

    def fixThreshold(self):
        '''
        Return the fixed threshold for this subaperture
        '''
        return self._fixThreshold

    def powerCoeff(self):
        '''
        Return the power coefficient for this subaperture
        '''
        return self._powerCoeff

    def linearCoeff(self):
        '''
        Return the "gamma" linearization coefficient
        '''
        return self._linearCoeff

    # def pupilCoords(self):
    #     '''
    #     Return coordinates of subaperture's center in the pupil
    #     The center of the pupil is (0,0)
    #     N= (0,1), S=(0,-1), E=(1,0), W=(-1,0)
    #     '''
    #     return self._pupilCoord

    def normalizationCoeff(self):
        return self._normalizationCoeff

    def setPixelWeight(self, wvec):
        '''
        Set the pixel weight mask. wvec should must be an array/list
        of length size**2
        '''
        wvec = np.asarray(wvec).flatten()
        assert (self._size * self._size,) == wvec.shape
        self._pixelWeight = wvec

    def setMaxGain(self, gain):
        '''
        Set the adaptive threshold for this subaperture
        '''
        self._maxGain = gain

    def setFixThreshold(self, th):
        '''
        Set the fixed threshold for this subaperture
        '''
        self._fixThreshold = th

    def setPowerCoeff(self, pw):
        '''
        Set the power coefficient for this subaperture
        '''
        self._powerCoeff = pw

    def setLinearCoeff(self, lc):
        '''
        Set the "gamma" linearization coefficient
        '''
        self._linearCoeff = lc

    # def setPupilCoords(self, pupilCoord):
    #     self._pupilCoord = pupilCoord

    def setNormalizationCoeff(self, normC):
        self._normalizationCoeff = normC

    def dump(self, p=True):
        '''
        Dump contents of subaperture
        '''
        dump = {}
        dump['ID'] = self.ID()
        dump['size'] = self.size()
        dump['pixelList'] = self.pixelList()
        dump['pixelWeight'] = self.pixelWeight()
        dump['pixelMask'] = self.pixelMask()
        dump['weightMask'] = self.weightMask()
        dump['maxGain'] = self.maxGain()
        dump['fixThreshold'] = self.fixThreshold()
        dump['powerCoeff'] = self.powerCoeff()
        dump['linearCoeff'] = self.linearCoeff()
        # dump['pupilCoord'] = self.pupilCoords()
        dump['normalizationCoeff'] = self.normalizationCoeff()
        if p:
            print(dump)
        return dump

    # def centerInCCDCoordinates(self):
    #     return np.array((self.pixelList()[36] % 264,
    #                      self.pixelList()[36] / 264))

    @staticmethod
    def createSubap(ID, detector_shape, subaperture_size, bl):
        pxidx = np.arange(
            detector_shape[1] * detector_shape[0]
        ).reshape(detector_shape[1], detector_shape[0])
        npixpersub = subaperture_size
        mask = pxidx[bl[0]:bl[0] + npixpersub, bl[1]:bl[1] + npixpersub]
        subap = ShSubaperture(ID, mask.flatten(), detector_shape,
                              subaperture_size)
        return subap


class ShSubapertureSet(dict):
    '''
    A subaperture set is a dictionary of subaperture object.
    Can save/restore itself.
    '''

    def __init__(self):
        pass

    def addSubap(self, subapObj):
        if subapObj.ID() in self.keys():
            raise ValueError
        self[subapObj.ID()] = subapObj

    def removeSubap(self, subapID):
        if isinstance(subapID, numbers.Number):
            if subapID in self:
                del self[subapID]
        else:
            for idx in subapID:
                if idx in self:
                    del self[idx]

    def update_fix_threshold(self, threshold):
        for i in self.values():
            i.setFixThreshold(threshold)

    def shiftSubap(self, subapID, deltaXY):
        '''
        subapID is a vector of subapIDs or a scalar.
        DeltaXY is a 2 elements numpy array
        '''
        if isinstance(subapID, numbers.Number):
            if subapID in self:
                self._shiftSA(subapID, deltaXY)
        else:
            for idx in subapID:
                if idx in self:
                    self._shiftSA(idx, deltaXY)

    def _shiftSA(self, subapID, deltaXY):
        pixelList = self[subapID].pixelList()
        det_shape = (self[subapID].ccdy, self[subapID].ccdx)
        suba_size = self[subapID].size()
        # if(self[subapID].wfsID == 0):
        #     if(7 <= pixelList[7] % 264 + deltaXY[1] <= 263):
        #         pixelList += deltaXY[1]
        #     else:
        #         raise Exception('Subap %d shift %d,%d is not allowed' %
        #                     (subapID, deltaXY[0], deltaXY[1]))
        # else:
        dx = deltaXY[1]
        dy = deltaXY[0]
        ccdx = det_shape[1]
        ccdy = det_shape[0]
        if (
            (suba_size-1 <= pixelList[suba_size-1] % ccdx + dx <= ccdx-1) &
            (0 <= pixelList[0] / ccdy + dy <= ccdy - 1 - suba_size)
        ):
            pixelList += dx
            pixelList += ccdy * dy
            # if((7 <= pixelList[7] % 264 + deltaXY[1] <= 263) &
            #    (0 <= pixelList[0] / 264 + deltaXY[0] <= 263 - 8)):
            #     pixelList += deltaXY[1]
            #     pixelList += 264 * deltaXY[0]
        else:
            raise Exception('Subap %d shift %d,%d is not allowed' %
                            (subapID, deltaXY[0], deltaXY[1]))

    # def getSubaperturesPerWFSID(self, wfsID):
    #     return [k for k, v in self.items() if v.wfsID() == wfsID]

    # def _checkShiftPupil(self, subapID, deltaXY):
    #     for s in subapID:
    #         pl = self[s].pixelList()
    #         if((0 <= pl[0] % 264 + deltaXY[1] <= 263 - 8) &
    #            (0 <= pl[0] / 264 + deltaXY[0] <= 263 - 8)):
    #             continue
    #         else:
    #             return False
    #     return True

    # def shiftPupil(self, pupil, deltaXY):
    #     subapID = self.getSubaperturesPerWFSID(pupil)
    #     if(pupil == 0):
    #         deltaXY[0] = 0
    #     if(self._checkShiftPupil(subapID, deltaXY)):
    #         for s in subapID:
    #             pixelList = self[s].pixelList()
    #             pixelList += deltaXY[1]
    #             pixelList += 264 * deltaXY[0]
    #     else:
    #         raise Exception('Pupil %d shift %d,%d is outside boundaries' %
    #                         (pupil, deltaXY[0], deltaXY[1]))

    # def numberOfSubaperturesPerWFSID(self, wfsID):
    #     return len(self.getSubaperturesPerWFSID(wfsID))

    # def centerOfPupilInCCDCoordinates(self, wfsID):
    #     subset = self.getSubaperturesPerWFSID(wfsID)
    #     coord = np.array([self[subapID].pupilCoords() for subapID in subset])
    #     id0 = np.sum((coord - np.tile([0.5, 0.5], (coord.shape[0], 1))) ** 2,
    #                 axis=1).argmin()
    #     id1 = np.sum((coord - np.tile(-coord[id0], (coord.shape[0], 1))) ** 2,
    #                 axis=1).argmin()
    #     tr = self[subset[id0]].centerInCCDCoordinates()
    #     bl = self[subset[id1]].centerInCCDCoordinates()
    #     return (tr + bl) / 2

    def save(self, filename, header, overwrite=False):

        ID = np.squeeze(np.dstack([x.ID() for x in self.values()]))
        # WfsID = np.squeeze(np.dstack([x.wfsID() for x in self.values()]))
        det_shape = np.squeeze(np.dstack([(x.ccdy, x.ccdx)
                                          for x in self.values()]))
        suba_size = np.squeeze(np.dstack([x.size() for x in self.values()]))
        px = np.squeeze(np.dstack([x.pixelList() for x in self.values()]))
        we = np.squeeze(np.dstack([x.pixelWeight() for x in self.values()]))
        fixT = np.squeeze(np.dstack([x.fixThreshold() for x in self.values()]))
        maxG = np.squeeze(np.dstack([x.maxGain() for x in self.values()]))
        powC = np.squeeze(np.dstack([x.powerCoeff() for x in self.values()]))
        linC = np.squeeze(np.dstack([x.linearCoeff() for x in self.values()]))
        # pupC = np.squeeze(np.dstack([x.pupilCoords() for x in self.values()]))
        normC = np.squeeze(np.dstack([x.normalizationCoeff()
                                     for x in self.values()]))

        if os.path.isfile(filename):
            if overwrite == True:
                os.remove(filename)
            else:
                raise IOError('File %s exists. Use overwrite' % filename)

        fits.append(filename, ID, header)
        # fits.append(filename, WfsID)
        fits.append(filename, px)
        fits.append(filename, det_shape)
        fits.append(filename, suba_size)
        fits.append(filename, we)
        fits.append(filename, fixT)
        fits.append(filename, maxG)
        fits.append(filename, powC)
        fits.append(filename, linC)
        # fits.append(filename, pupC)
        fits.append(filename, normC)

    @staticmethod
    def restore(filename):
        '''
        Restore subapertures definition,
        and return a SubapertureSet object
        '''
        hdulist = fits.open(filename)
        ID = hdulist[0].data
        # WfsID = hdulist[1].data
        px = hdulist[1].data
        det_shape = hdulist[2].data
        suba_size = hdulist[3].data
        we = hdulist[4].data
        fixT = hdulist[5].data
        maxG = hdulist[6].data
        powC = hdulist[7].data
        linC = hdulist[8].data
        # try:
        #     pupC = hdulist[8].data
        # except:
        #     pupC = np.zeros((2, len(ID)))
        try:
            normC = hdulist[9].data
        except:
            normC = np.zeros(len(ID))
        hdulist.close()

        subapSet = ShSubapertureSet()
        nsubap = px.shape[1]
        for i in range(nsubap):
            s = ShSubaperture(ID[i],
                              # WfsID[i],
                              px[:, i],
                              det_shape[:, i],
                              suba_size[i],
                              we[:, i],
                              fixT[i],
                              maxG[i],
                              powC[i],
                              linC[i],
                              # pupC[:, i],
                              normC[i])
            subapSet.addSubap(s)
        return subapSet

    # @staticmethod
    # def create(wfsTag, cc, nsub, appendTo=None):
    #     '''
    #     wfsTag:    integer (typ 0=BLUE, 1=YELLOW, 2=RED)
    #     cc:        int[2], coordinate of center of the pupil in the ccd
    #                        reference system (y, x)
    #     nsub:              int, number of subapertures on a diameter
    #     appendTo:          SubapertureSet to whom append new subapertures
    #     '''
    #     pxidx = np.arange(264 * 264).reshape(264, 264).T
    #     npixpersub = 8  # TODO: get all these 8 and 264 from BCUConstants
    #     pupil_pos_vect = np.linspace(-(nsub - 1.) / nsub, (nsub - 1.) / nsub,
    #                                 nsub)
    #     if appendTo == None:
    #         subapSet = ShSubapertureSet()
    #     else:
    #         subapSet = copy.copy(appendTo)
    #     bl = [cc[0] - npixpersub * nsub / 2, cc[1] - npixpersub * nsub / 2]
    #     for x in range(0, nsub):
    #         for y in range(0, nsub):
    #             pupil_pos = pupil_pos_vect[[x, y]]
    #             if np.linalg.norm(pupil_pos) > (1):  # +0.7/nsub):
    #             # if np.linalg.norm(pupil_pos) > (1+0.7/nsub):
    #                 continue
    #             mask = pxidx[bl[0] + y * npixpersub:
    #                          bl[0] + (y + 1) * npixpersub,
    #                          bl[1] + x * npixpersub:
    #                          bl[1] + (x + 1) * npixpersub]
    #             weights = np.ones((npixpersub, npixpersub))
    #             subapSet.addSubap(
    #                         ShSubaperture(len(subapSet) + 1,
    #                                     wfsTag,
    #                                     mask.flatten('F'),
    #                                     weights))
    #                                     # ,
    #                                     # pupilCoords=pupil_pos))
    #     return subapSet

    @staticmethod
    def createMinimalSet(ID_list, detector_shape, subaperture_size, bl_list):

        def createSingleSubap(ID, detector_shape, subaperture_size, bl):
            pxidx = np.arange(
                detector_shape[1] * detector_shape[0]
            ).reshape(detector_shape[1], detector_shape[0])
            npixpersub = int(subaperture_size)
            mask = pxidx[int(bl[0]): int(bl[0] + npixpersub),
                         int(bl[1]): int(bl[1] + npixpersub)]
            subap = ShSubaperture(ID, mask.flatten(), detector_shape,
                                  npixpersub)
            return subap

        sset = ShSubapertureSet()
        for i in range(len(ID_list)):
            sset.addSubap(
                createSingleSubap(ID_list[i],
                                  detector_shape,
                                  subaperture_size,
                                  bl_list[i]))
        return sset
