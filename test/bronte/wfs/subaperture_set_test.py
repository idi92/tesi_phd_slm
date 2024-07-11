#!/usr/bin/env python

import unittest
# from mockito import mock, when
from astropy.io.fits import Header
import numpy as np
from bronte.wfs.subaperture_set import ShSubapertureSet, ShSubaperture


class ShSubapertureTest(unittest.TestCase):

    def setUp(self):
        # self._size= BCUConstants.subapSize
        self._size = 10
        self._ID = 0
        # self._wfsID= mock()
        self._idxPixelList = np.arange(self._size * self._size)
        self._detector_shape = (500, 500)
        self._pixelWeight = np.ones(self._size * self._size)
        self._fixThreshold = 0.0
        self._maxGain = 0.0
        self._powerCoeff = 1.0
        self._linearCoeff = 1.

    def _makeSubap(self):
        subap = ShSubaperture(self._ID, self._idxPixelList,
                              self._detector_shape, self._size,
                              self._pixelWeight, self._fixThreshold,
                              self._maxGain, self._powerCoeff,
                              self._linearCoeff)
        return subap

    def test_equal(self):
        self._ID=3
        subap0= self._makeSubap()
        subap1= self._makeSubap()
        self.assertEqual(subap0, subap1)
        
    def test_ID(self):
        self._ID=3
        subap= self._makeSubap()
        self.assertEqual(subap.ID(), self._ID)

    def test_ID_is_not_none(self):
        self._ID=None
        self.assertRaises(ValueError, self._makeSubap)

    def test_size(self):
        subap= self._makeSubap()
        self.assertEqual(subap.size(), self._size)

    def test_pixelList(self):
        self._idxPixelList= np.arange(self._size* self._size)
        subap= self._makeSubap()
        assert(np.all(subap.pixelList()== self._idxPixelList))

    def test_pixelList_len(self):
        self._idxPixelList= np.arange(self._size*self._size-1)
        self.assertRaises(AssertionError, self._makeSubap)

    def test_pixelWeight(self):
        self._pixelWeight= np.arange(self._size* self._size)*0.1
        subap= self._makeSubap()
        self.assertEqual(subap.pixelWeight().shape, (self._size**2,))
        assert(np.all(subap.pixelWeight()== self._pixelWeight))

    def test_pixelWeight_len(self):
        self._pixelWeight= np.arange(self._size*self._size-1)
        self.assertRaises(AssertionError, self._makeSubap)

    def test_pixelWeight_default(self):
        subap=ShSubaperture(self._ID, self._idxPixelList, self._detector_shape,
                            self._size)
        assert(np.all(subap.pixelWeight()== np.ones(self._size**2)))

    def test_pixelMask(self):
        subap= self._makeSubap()
        pxmask=subap.pixelMask()
        self.assertEqual(pxmask.shape,
                self._detector_shape)
        self.assertTrue(np.min(pxmask) >= 0)
        self.assertTrue(np.max(pxmask) <= self._size**2)
        
#TODO: fix this
    # def test_pixelMask_particular(self):
    #     ccdy=self._detector_shape[0]
    #     ccdx=self._detector_shape[1]
    #     pxidx=np.arange(ccdx*ccdy).reshape(ccdy, ccdx)
    #     mask=pxidx[2:10, 244:252]
    #     self._idxPixelList = mask.flatten()
    #     subap= self._makeSubap()
    #     pxmask=subap.pixelMask()
    #     self.assertTrue(np.all((pxmask[2:10, 244:252]-mask) != 0))

    def test_weightMask(self):
        subap= self._makeSubap()
        wtmask=subap.weightMask()
        self.assertEqual(wtmask.shape,
                self._detector_shape)
        self.assertTrue(np.min(wtmask) >= 0)

#TODO: fix this
    # def test_weightMask_particular(self):
    #     ccdy=self._detector_shape[0]
    #     ccdx=self._detector_shape[1]
    #     pxidx=np.arange(ccdx*ccdy).reshape(ccdy, ccdx)
    #     self._idxPixelList= pxidx[2:10, 244:252].flatten()
    #     self._pixelWeight= np.arange(self._size* self._size)*0.1
    #     subap= self._makeSubap()
    #     wemask=subap.weightMask()
    #     self.assertAlmostEqual(wemask[2, 244], 0)
    #     self.assertAlmostEqual(wemask[3, 244], 0.1)
    #     self.assertAlmostEqual(wemask[9, 251], 6.3)


    def test_columnMajorFlipIdx_particular(self):
        pass

    def test_maxGain(self):
        self._maxGain= 3.
        subap= self._makeSubap()
        self.assertEqual(subap.maxGain(), self._maxGain)

    def test_maxGain_default(self):
        subap=ShSubaperture(self._ID, self._idxPixelList, self._detector_shape,
                            self._size)
        self.assertEqual(subap.maxGain(), 0.)

    def test_fixThreshold(self):
        self._fixThreshold= 3.
        subap= self._makeSubap()
        self.assertEqual(subap.fixThreshold(), self._fixThreshold)

    def test_fixThreshold_default(self):
        subap=ShSubaperture(self._ID, self._idxPixelList, self._detector_shape,
                            self._size)
        self.assertEqual(subap.fixThreshold(), 0.)

    def test_linearCoeff(self):
        self._linearCoeff= 1.5
        subap= self._makeSubap()
        self.assertEqual(subap.linearCoeff(), self._linearCoeff)

    def test_linearCoeff_default(self):
        subap=ShSubaperture(self._ID, self._idxPixelList, self._detector_shape,
                            self._size)
        self.assertEqual(subap.linearCoeff(), 1)

    def test_setPixelWeight(self):
        ww= np.arange(self._size* self._size)*0.1
        subap= self._makeSubap()
        subap.setPixelWeight(ww)
        self.assertTrue(np.all(subap.pixelWeight()== ww))

    def test_setMaxGain(self):
        subap= self._makeSubap()
        subap.setMaxGain(0.3)
        self.assertEqual(subap.maxGain(), 0.3)

    def test_setFixThreshold(self):
        subap= self._makeSubap()
        subap.setFixThreshold(0.1)
        self.assertEqual(subap.fixThreshold(), 0.1)

    def test_setPowerCoeff(self):
        subap= self._makeSubap()
        subap.setPowerCoeff(1.5)
        self.assertEqual(subap.powerCoeff(), 1.5)

    def test_setLinearCoeff(self):
        subap= self._makeSubap()
        subap.setLinearCoeff(3.0)
        self.assertEqual(subap.linearCoeff(), 3.0)


    def test_setNormalizationCoeff(self):
        subap= self._makeSubap()
        subap.setNormalizationCoeff(1)
        self.assertEqual(subap.linearCoeff(), 1)


    def test_dump(self):
        subap= self._makeSubap()
        dump= subap.dump()
        self.assertEqual(dump['ID'], subap.ID())
        # self.assertEqual(dump['WfsID'], subap.wfsID())


class FakeSubaperture():
    
    def __init__(self, i):
        self._i = i
        
    def ID(self):
        return self._i + 10 


class ShSubapertureSetTest(unittest.TestCase):

    def setUp(self):
        pass

    def _makeSubapList(self, nitem):
        slist=[]
        for i in range(nitem):
            # subap= mock()
            # when(subap).ID().thenReturn(10+i)
            # when(subap).foo().thenReturn(100+i)
            subap = FakeSubaperture(i)
            slist.append(subap)
        return slist

    def test_addSubap(self):
        slist= self._makeSubapList(200)
        sset= ShSubapertureSet()
        for subap in slist:
            sset.addSubap(subap)
        # for k in sset.keys():
        #     self.assertEqual(sset[k].foo(), 90+k)

    def test_addSubap_unique(self):
        slist= self._makeSubapList(2)
        sset= ShSubapertureSet()
        for subap in slist:
            sset.addSubap(subap)
        self.assertRaises(ValueError, sset.addSubap, slist[0])

    def test_removeSubap_scalar(self):
        slist= self._makeSubapList(3)
        sset= ShSubapertureSet()
        for subap in slist:
            sset.addSubap(subap)
        sset.removeSubap(11)

        def key11():
            return sset[11]
        self.assertRaises(KeyError, key11)

    def test_removeSubap_vector(self):
        slist= self._makeSubapList(10)
        sset= ShSubapertureSet()
        for subap in slist:
            sset.addSubap(subap)
        removeList= range(10, 10+len(slist), 2)
        sset.removeSubap(removeList)
        leftList = np.array(list(sset.keys()))
        self.assertTrue(np.all(leftList==range(11, 10+len(slist), 2)))

    def test_save_and_restore(self):
        filename='/tmp/testsubapset.fits'
        hdr= Header()
        ID_list = [0, 1]
        det_shape = (50, 50)
        suba_size = 5
        bl_list= [(10, 10), (35, 35)]
        sset=ShSubapertureSet.createMinimalSet(
            ID_list, det_shape, suba_size, bl_list)
        sset.save(filename, hdr, True)
        restored= ShSubapertureSet.restore(filename)
        for i in sset.keys():
            self.assertEqual(sset[i], restored[i])


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
