import numpy as np
import numpy.ma as ma
from scipy import ndimage
from arte.utils.decorator import logEnterAndExit
import logging

class PCSlopeComputer():

    _X_SLOPES_IDX = 0
    _Y_SLOPES_IDX = 1

    '''
    Implements a Slope Computer on PC using the same algorithm of the RTC
    This class retrieve subapertures definition from a
    ShSubapertureSet object.

    Compute maps to visualize subapertures
    properties: location on the CCD, pixel weights, subaperture flux.
    Useful for debugging of RTC, visualization of pupil map and
    during the definition of the valid subapertures
    '''

    def __init__(self, subapertureSet):
        self._logger = logging.getLogger('PCSlopeComputer')
        self.set_subapertures(subapertureSet)
        self.CCD_HEIGHT_IN_PIXEL = list(subapertureSet.values())[0].ccdy
        self.CCD_WIDTH_IN_PIXEL = list(subapertureSet.values())[0].ccdx
        self._threValidSubap = 0.3
        self._background = np.zeros((self.CCD_HEIGHT_IN_PIXEL,
                              self.CCD_WIDTH_IN_PIXEL))
        self._flat = np.ones(
            (self.CCD_HEIGHT_IN_PIXEL, self.CCD_WIDTH_IN_PIXEL))
        self._common_mode = np.zeros((self.CCD_HEIGHT_IN_PIXEL,
                                     self.CCD_WIDTH_IN_PIXEL))
        self._common_mode_threshold = 1000000
        self.upload_raw_frame(np.zeros((self.CCD_HEIGHT_IN_PIXEL,
                                        self.CCD_WIDTH_IN_PIXEL)))
        self._slopesNull = np.zeros(len(subapertureSet) * 2)

    def _reset_all_computed_attributes(self):
        # self._spotdict = None
        # self._spotarr = None
        self._slopesarr = None
        self._slopesdict = None
        self._momentdict = None

    @property
    def subapertures(self):
        return self._subapSet

    def set_subapertures(self, subapertureSet):
        try:
            if len(self._subapSet) != len(subapertureSet):
                self._slopesNull = np.zeros(len(subapertureSet) * 2)
        except AttributeError:
            pass
        self._subapSet = subapertureSet
        self._reset_all_computed_attributes()
        self._update_number_of_subaps_in_pupil()
        self._getSubapertureSize()

    def _update_number_of_subaps_in_pupil(self):
        self._numberOfSubapsInPupil = np.zeros(1)
        for i in self._subapSet.values():
            self._numberOfSubapsInPupil += 1

    def _getSubapertureSize(self):
        '''
        Assume all subapertures have the same size.
        '''
        # TODO check that every subaperture has the same size
        self._subaperture_size_in_px = list(self._subapSet.values())[0].size()

    @property
    def subaperture_size(self):
        return self._subaperture_size_in_px

    def number_of_subaps_in_pupil(self):
        return self._numberOfSubapsInPupil

    def remove_low_flux_subaps(self, threshold=None):
        # WARNING: non idempotent
        fluxDict = self.subapertures_flux_dictionary()
        medianFlux = np.median(list(fluxDict.values()))
        if threshold is None:
            threshold = self._threValidSubap * medianFlux
        keylow = [key for (key, value) in fluxDict.items() if
                  value < threshold]
        self._subapSet.removeSubap(keylow)
        self._reset_all_computed_attributes()
        self._update_number_of_subaps_in_pupil()

    def remove_high_rms_slopes_subaps(self, frames):
        numSubap = self.total_number_of_subapertures()
        nIter = frames.shape[2]
        # slopesx=np.zeros((nIter, numSubap))
        # slopesy=np.zeros((nIter, numSubap))
        slopesTime = np.zeros((nIter, numSubap, 2))
        for i in range(nIter):
            self.set_frame(frames[:, :, i])
            slopesDict = self.slopes_dictionary()
            IDs = np.array(list(slopesDict.keys()))
            slopesTime[i] = np.array(list(slopesDict.values()))
            # slopesx[i], IDs= self.slopeComputerDevice.onPC.slopesX()
            # slopesy[i], IDs= self.slopeComputerDevice.onPC.slopesY()
        rmsx = np.std(slopesTime[:, :, self._X_SLOPES_IDX], axis=0)
        rmsy = np.std(slopesTime[:, :, self._Y_SLOPES_IDX], axis=0)
        badSubap = np.nonzero((rmsx > np.median(rmsx) * 3) |
                              (rmsy > np.median(rmsy) * 3))[0]
        self._subapSet.removeSubap(np.array(IDs)[badSubap])
        self._reset_all_computed_attributes()
        self._update_number_of_subaps_in_pupil()
        return badSubap, IDs

    def frame(self):
        return self._frame

    def set_frame(self, frame):
        self._frame = frame
        self._reset_all_computed_attributes()

    def upload_raw_frame(self, rawFrame):
        self._rawframe = rawFrame
        self.set_frame(self.compute_processed_frame(self._rawframe))

    def upload_background_frame(self, backgroundFrame):
        self._background = backgroundFrame

    def upload_common_mode_map(self, cmmap):
        self._common_mode = cmmap

    def upload_flat_frame(self, flatFrame):
        self._flat = flatFrame

    def set_common_mode_threshold(self, threshold):
        self._common_mode_threshold = threshold

    def compute_processed_frame(self, rawima):
        self._background_corrected = (rawima.astype(np.int32) -
                                      self._background.astype(np.int32)).astype(np.float32)
        # cm_correction= self.common_mode_correction(
        #                 self._background_corrected, self._common_mode,
        #                 self._common_mode_threshold)
        # self._cm_corrected= self._background_corrected - cm_correction
        processedFrame = self._background_corrected / self._flat
        return processedFrame

    def frame_correction_info(self):
        ret = {}
        ret['raw_frame'] = self._rawframe
        ret['background_corrected'] = self._background_corrected
        # ret['cm_corrected']= self._cm_corrected
        ret['flat_corrected'] = self._frame
        return ret

    def subapertures_flux_dictionary(self):
        fluxDict = dict()
        flattenFrame = self._frame.flatten()
        for i in self._subapSet.values():
            px_values = flattenFrame[i.pixelList()]
            fluxDict[i.ID()] = px_values.sum()
        return fluxDict

    def total_number_of_subapertures(self):
        return len(self._subapSet)

    def subapertures_flux_map(self):
        fluxSubapsFrame = np.zeros(self.frame().size, dtype=float)
        flattenFrame = self._frame.flatten()
        for i in self._subapSet.values():
            px_values = flattenFrame[i.pixelList()]
            fluxSubapsFrame[i.pixelList()] = px_values.sum()
        return fluxSubapsFrame.reshape(self.frame().shape)

    def flux_in_pupil(self):
        flattenFrame = self._frame.flatten()
        fluxSubapsTot = []
        for i in self._subapSet.values():
            px_values = flattenFrame[i.pixelList()]
            fluxSubap = px_values.sum()
            fluxSubapsTot.append(fluxSubap)
        fluxSubapsMean = np.array(fluxSubapsTot
                                  ).sum() / self._numberOfSubapsInPupil
        return fluxSubapsMean

    def subapertures_id_map(self):
        frame = np.zeros(self.frame().size, dtype=float)
        for i in self._subapSet.values():
            frame[i.pixelList()] = i.ID()
        return frame.reshape(self.frame().shape)

    def subapertures_pixels_map(self):
        frame = np.zeros(self.frame().size, dtype=float)
        flattenFrame = self._frame.flatten()
        for i in self._subapSet.values():
            px_values = flattenFrame[i.pixelList()]
            frame[i.pixelList()] = px_values
        return frame.reshape(self.frame().shape)

    def subapertures_weights_map(self):
        frame = np.zeros(self.frame().size, dtype=float)
        for i in self._subapSet.values():
            frame[i.pixelList()] = i.pixelWeight()
        return frame.reshape(self.frame().shape)

    def slopes_x_map(self):
        frame = np.zeros(self.frame().size, dtype=float)
        for i in self._subapSet.values():
            frame[i.pixelList()] = self.slopes_dictionary()[
                i.ID()][self._X_SLOPES_IDX]
        return frame.reshape(self.frame().shape)

    def slopes_y_map(self):
        frame = np.zeros(self.frame().size, dtype=float)
        for i in self._subapSet.values():
            frame[i.pixelList()] = self.slopes_dictionary()[
                i.ID()][self._Y_SLOPES_IDX]
        return frame.reshape(self.frame().shape)

    # def xCoordinatesSubapsMap(self):
    #     subapsMap= np.zeros(self.frame().size, dtype=float)
    #     for i in self._subapSet.values():
    #         subapsMap[i.pixelList()] = i.pupilCoords()[0]
    #     return subapsMap.reshape(self.frame().shape)
    #
    #
    # def yCoordinatesSubapsMap(self):
    #     subapsMap= np.zeros(self.frame().size, dtype=float)
    #     for i in self._subapSet.values():
    #         subapsMap[i.pixelList()] = i.pupilCoords()[1]
    #     return subapsMap.reshape(self.frame().shape)

    # def spotsFitFWHMMap(self):
    #     subapsMap= np.zeros(self.frame().size, dtype=float)
    #     dicto= self.spotsFitDictionary()
    #     for k, v in self._subapSet.items():
    #         # subapsMap[v.pixelList()] = np.sqrt((dicto[k][3] ** 2 +
    #         #                                    dicto[k][4] ** 2) / 2.) * 2.335
    #         subapsMap[v.pixelList()] = np.sqrt(dicto[k][3] *
    #                                            dicto[k][4])
    #     return subapsMap.reshape(self.frame().shape)

    def spots_second_moment_map(self):
        subapsMap = np.zeros(self.frame().size, dtype=float)
        dicto = self.spots_image_moment_dictionary()
        for k, v in self._subapSet.items():
            subapsMap[v.pixelList()] = \
                dicto[k].equivalentCircularGaussianSpotSigma()
        return subapsMap.reshape(self.frame().shape)

    def spots_moment_eccentricity_map(self):
        subapsMap = np.zeros(self.frame().size, dtype=float)
        dicto = self.spots_image_moment_dictionary()
        for k, v in self._subapSet.items():
            subapsMap[v.pixelList()] = dicto[k].eccentricity()
        return subapsMap.reshape(self.frame().shape)

    def spots_moment_orientation_map(self):
        subapsMap = np.zeros(self.frame().size, dtype=float)
        dicto = self.spots_image_moment_dictionary()
        for k, v in self._subapSet.items():
            subapsMap[v.pixelList()] = dicto[k].orientation() * 180. / np.pi
        return subapsMap.reshape(self.frame().shape)

    # def spotsFitEllipticityMap(self):
    #     subapsMap= np.zeros(self.frame().size, dtype=float)
    #     dicto= self.spotsFitDictionary()
    #     for k, v in self._subapSet.items():
    #         subapsMap[v.pixelList()] = np.abs(1 - dicto[k][3] / dicto[k][4])
    #     return subapsMap.reshape(self.frame().shape)
    #
    #
    # def spotsFitOrientationMap(self):
    #     subapsMap= np.zeros(self.frame().size, dtype=float)
    #     dicto= self.spotsFitDictionary()
    #     for k, v in self._subapSet.items():
    #         subapsMap[v.pixelList()] = dicto[k][5] * 180. / np.pi
    #     return subapsMap.reshape(self.frame().shape)

    def subapertures_map(self):
        sf = np.zeros(self.frame().size, dtype=float)
        for i in self._subapSet.values():
            pl = i.pixelList()
            sz = int(self.subaperture_size)
            sf[pl[0:sz]] = 1
            sf[pl[sz*(sz-1):]] = 1
            sf[pl[sz:sz*(sz-1):sz]] = 1
            sf[pl[2*sz-1:sz*(sz-1):sz]] = 1
        return sf.reshape(self.frame().shape)

    @logEnterAndExit("Computing slopes", "Slopes computed", level='debug')
    def _compute_slopes(self):
        self._slopesdict = {}

        (sy, sx) = (self.subaperture_size, self.subaperture_size)
        x = np.linspace(-1 + 1. / sx, 1 - 1. / sx, sx)
        y = np.linspace(-1 + 1. / sy, 1 - 1. / sy, sy)
        xm = np.repeat(x[np.newaxis, :], sy, axis=0).flatten()
        ym = np.repeat(y[:, np.newaxis], sx, axis=1).flatten()
        flattenFrame = self._frame.flatten()
        for i in self._subapSet.values():
            pixelList = flattenFrame[i.pixelList()]
            threshold = i.fixThreshold() + i.maxGain() * np.max(pixelList)
            px_values = pixelList - threshold
            px_values = np.clip(px_values, 0, 1e10)
            normalize = (px_values * i.pixelWeight()).sum()
            if np.all(i.pixelWeight() == 0):
                xc = 0.
                yc = 0.
            else:
                xc = (px_values * i.pixelWeight() * xm).sum() / normalize
                yc = (px_values * i.pixelWeight() * ym).sum() / normalize

            xc = 0. if np.isnan(xc) else xc
            yc = 0. if np.isnan(yc) else yc

            self._slopesdict[i.ID()] = np.array([xc, yc])
        self._slopesarr = np.array(list(self._slopesdict.values()))

    def central_moment(self, mr, ox, oy):
        di = mr.shape[0]
        ddi = di/2-0.5
        ll = np.linspace(-ddi, ddi, di)
        xl, yl = np.meshgrid(ll, ll)
        norm = np.sum(mr)
        y0, x0 = ndimage.measurements.center_of_mass(mr)
        return np.sum(mr*((-x0+xl+ddi)**ox)*((-y0+yl+ddi)**oy))/norm

    def _compute_image_moment(self):
        from arte.utils.image_moments import ImageMoments
        self._compute_slopes()
        self._momentdict = {}

        flattenFrame = self._frame.flatten()

        for i in self._subapSet.values():
            # px_values= flattenFrame[i.pixelList()] - \
            #     Constants.READ_OUT_NOISE_IN_ADU
            px_values = flattenFrame[i.pixelList()]
            px_values = np.clip(px_values, 0, 1e10)
            subap = px_values.reshape(
                (self.subaperture_size, self.subaperture_size))
            moments = ImageMoments(subap)

#            amplitude= moments.central_moment(0, 0)
#            cen= moments.centroid()
#            x_mean= cen[0]
#            y_mean= cen[1]
#            x_width= moments.centralNormalizedMoment(2, 0)
#            y_width= moments.centralNormalizedMoment(0, 2)
#            eccentricity= moments.eccentricity()
#            orientation= moments.orientation()

            self._momentdict[i.ID()] = moments

#            self._momentdict[i.ID()]=np.array([amplitude,
#                                               x_mean,
#                                               y_mean,
#                                               x_width,
#                                               y_width,
#                                               eccentricity,
#                                               orientation])

#        self._secondarr = np.asarray(np.sqrt(
#                                    np.array(self._momentdict.values())[:, 3]*
#                                    np.array(self._momentdict.values())[:, 4])
#                                   ).flatten()

    def slopes(self):
        if self._slopesarr is None:
            self._compute_slopes()
        return self._slopesarr

    def slopes_dictionary(self):
        if self._slopesdict is None:
            self._compute_slopes()
        return self._slopesdict


#    def spotsSecondMoment(self):
#        if self._secondarr is None:
#            self._compute_image_moment()
#        return self._secondarr

    # def spotsSecondMomentPerPupil(self):
    #     if self._momentdict is None:
    #         self._compute_image_moment()
    #     datadict= self.spotsImageMomentDictionaryPerPupil()
    #     blue= self._compute_spot_size_with_second_moment_per_pupil(
    #         datadict[LaserBeam.Blue])
    #     red= self._compute_spot_size_with_second_moment_per_pupil(
    #         datadict[LaserBeam.Red])
    #     yellow= self._compute_spot_size_with_second_moment_per_pupil(
    #         datadict[LaserBeam.Yellow])
    #     return {LaserBeam.Blue: blue,
    #             LaserBeam.Yellow: yellow,
    #             LaserBeam.Red: red}


    def _compute_spot_size_with_second_moment_per_pupil(self, momentsDict):
        dummy = np.zeros(len(momentsDict))
        idx = 0
        for imageMoment in momentsDict.values():
            dummy[idx] = imageMoment.equivalentCircularGaussianSpotSigma()
            idx += 1
        return np.mean(dummy)

    # def spotsMomentEccentricityPerPupil(self):
    #     if self._momentdict is None:
    #         self._compute_image_moment()
    #     datadict= self.spotsImageMomentDictionaryPerPupil()
    #
    #     def computeEccentricity(datadict):
    #         covMat= self._compute_average_covariance_matrix_per_pupil(datadict)
    #         return self._compute_eccentricity_from_covariance_matrix(covMat)
    #
    #     blue= computeEccentricity(datadict[LaserBeam.Blue])
    #     yellow= computeEccentricity(datadict[LaserBeam.Yellow])
    #     red= computeEccentricity(datadict[LaserBeam.Red])
    #
    #     return {LaserBeam.Blue: blue,
    #             LaserBeam.Yellow: yellow,
    #             LaserBeam.Red: red}

    def _compute_average_covariance_matrix_per_pupil(self, momentsDict):
        dummy = np.zeros((len(momentsDict), 2, 2))
        idx = 0
        for imageMoment in momentsDict.values():
            dummy[idx] = imageMoment.covarianceMatrix()
            idx += 1
        return np.mean(dummy, axis=0)

    def _compute_eccentricity_from_covariance_matrix(self, covarianceMatrix):
        eigen = self._compute_eigenvalues_from_covariance_matrix(
            covarianceMatrix)
        return np.sqrt(1 - eigen[1] / eigen[0])

    def _compute_eigenvalues_from_covariance_matrix(self, covarianceMatrix):
        u11 = covarianceMatrix[1, 0]
        u20 = covarianceMatrix[0, 0]
        u02 = covarianceMatrix[1, 1]
        aa = 0.5 * (u20 + u02)
        bb = 0.5 * np.sqrt(4 * u11 ** 2 + (u20 - u02) ** 2)
        return np.array([aa + bb, aa - bb])

    def spots_image_moment_dictionary(self):
        if self._momentdict is None:
            self._compute_image_moment()
        return self._momentdict

    def slopes_std(self):
        # std di tutti i termini o lungo gli assi?
        return np.std(self.slopes())

    def slope_offset(self):
        return self._slopesNull

    def set_slope_offset(self, slopeOffset):
        self._slopesNull = slopeOffset

    def average_tip_tilt(self):
        return np.median(self.slopes(), axis=0)

    def common_mode_correction(self, frame, common_mode_map, threshold):
        # FOR ARGOS pnCCD
        half = self.CCD_HEIGHT_IN_PIXEL / 2
        q1 = PCSlopeComputer._half_common_mode_correction(
            frame[0:half, :], common_mode_map[0:half, :], threshold)
        frameToSubtract1 = np.tile(q1, half).reshape((half,
                                                      self.CCD_WIDTH_IN_PIXEL))
        q3 = PCSlopeComputer._half_common_mode_correction(
            frame[half:, :], common_mode_map[half:, :], threshold)
        frameToSubtract3 = np.tile(q3, half).reshape((half,
                                                      self.CCD_WIDTH_IN_PIXEL))
        res = np.zeros((self.CCD_HEIGHT_IN_PIXEL, self.CCD_WIDTH_IN_PIXEL))
        res[0:half, :] = frameToSubtract1
        res[half:, :] = frameToSubtract3
        return res

    @staticmethod
    def _column_common_mode_correction(frame, common_mode_map, threshold):
        masked_cm = ma.masked_equal(common_mode_map, 0)
        masked_frame = ma.array(frame, mask=ma.getmask(masked_cm))
        masked_frame_thre = ma.masked_greater(masked_frame, threshold)

        vv = masked_frame_thre.mean(axis=0).data
        return vv

    # def spotsFitFWHM(self):
    #     if self._spotarr is None:
    #         self._computeSpotsByFit()
    #     return self._spotarr

    # def spotsFitFWHMPerPupil(self):
    #     if self._momentdict is None:
    #         self._compute_image_moment()
    #     datadict= self.spotsFitDictionaryPerPupil()
    #     blue= np.median(np.sqrt(np.array(datadict[LaserBeam.Blue].
    #                                      values())[:, 3] *
    #                             np.array(datadict[LaserBeam.Blue].
    #                                      values())[:, 4]))
    #     yellow= np.median(np.sqrt(np.array(datadict[LaserBeam.Yellow].
    #                                        values())[:, 3] *
    #                               np.array(datadict[LaserBeam.Yellow].
    #                                        values())[:, 4]))
    #     red= np.median(np.sqrt(np.array(datadict[LaserBeam.Red].
    #                                     values())[:, 3] *
    #                            np.array(datadict[LaserBeam.Red].
    #                                     values())[:, 4]))
    #     return {LaserBeam.Blue: blue,
    #             LaserBeam.Yellow: yellow,
    #             LaserBeam.Red: red}

    # def spotsFitEllipticityPerPupil(self):
    #     if self._momentdict is None:
    #         self._compute_image_moment()
    #     datadict= self.spotsFitDictionaryPerPupil()
    #     blue= np.median(np.abs(1 - np.array(datadict[LaserBeam.Blue].
    #                                  values())[:, 3] /
    #                     np.array(datadict[LaserBeam.Blue].
    #                              values())[:, 4]))
    #     yellow= np.median(np.abs(1 - np.array(datadict[LaserBeam.Yellow].
    #                                    values())[:, 3] /
    #                       np.array(datadict[LaserBeam.Yellow].
    #                                values())[:, 4]))
    #     red= np.median(np.abs(1 - np.array(datadict[LaserBeam.Red].
    #                                 values())[:, 3] /
    #                    np.array(datadict[LaserBeam.Red].
    #                             values())[:, 4]))
    #     return {LaserBeam.Blue: blue,
    #             LaserBeam.Yellow: yellow,
    #             LaserBeam.Red: red}

    # def spotsFitDictionary(self):
    #     if self._spotdict is None:
    #         self._computeSpotsByFit()
    #     return self._spotdict

    # def spotsFitDictionaryPerPupil(self):
    #     if self._spotdict is None:
    #         self._computeSpotsByFit()
    #
    #     spotDict = dict()
    #     spotDict[LaserBeam.Blue]= dict()
    #     spotDict[LaserBeam.Yellow]= dict()
    #     spotDict[LaserBeam.Red]= dict()
    #
    #     for i in self._subapSet.values():
    #         if i.wfsID() == 0:
    #             spotDict[LaserBeam.Blue][i.ID()]= self._spotdict[i.ID()]
    #         elif i.wfsID() == 1:
    #             spotDict[LaserBeam.Yellow][i.ID()]= self._spotdict[i.ID()]
    #         elif i.wfsID() == 2:
    #             spotDict[LaserBeam.Red][i.ID()]= self._spotdict[i.ID()]
    #         else:
    #             raise Exception("Programming error")
    #     return spotDict

    # def _computeSpotsByFit(self):
    #     self._spotdict = {}
    #
    #     RON= 16.
    #     init_params=[100, 4, 4, 1., 1., 0]
    #     min_params=[0., 0., 0., 0.4, 0.4, - np.pi / 2.]
    #     max_params=[None, 8., 8., 4, 4, np.pi / 2.]
    #     stddev_err=np.ones((8, 8)) * RON
    #
    #     self.par= Parameters()
    #     self.par.add_many(
    #              ('amplitude', init_params[0], True, min_params[0], max_params[0]),
    #              ('x_mean', init_params[1], True, min_params[1], max_params[1]),
    #              ('y_mean', init_params[2], True, min_params[2], max_params[2]),
    #              ('x_stddev', init_params[3], True, min_params[3], max_params[3]),
    #              ('y_stddev', init_params[4], True, min_params[4], max_params[4]),
    #              ('theta', init_params[5], True, min_params[5], max_params[5]))
    #
    #     flattenFrame= self._frame.flatten()
    #
    #     for i in self._subapSet.values():
    #         px_values= flattenFrame[i.pixelList()]
    #         px_values= np.clip(px_values, 0, 1e10)
    #         subap= px_values.reshape((8, 8), order='F')
    #
    #         # reset initial parameters
    #         self.par['amplitude'].value= init_params[0]
    #         self.par['x_mean'].value= init_params[1]
    #         self.par['y_mean'].value= init_params[2]
    #         self.par['x_stddev'].value= init_params[3]
    #         self.par['y_stddev'].value= init_params[4]
    #         self.par['theta'].value= init_params[5]
    #
    #         # run lmfit minimize on subaperture
    #         self.res= self.fitGaussian2D(subap, stddev_err)
    #
    #         # extrat results
    #         self._spotdict[i.ID()] = np.array([self.res.params['amplitude'].value,
    #                                            self.res.params['x_mean'].value,
    #                                            self.res.params['y_mean'].value,
    #                                            self.res.params['x_stddev'].value * 2.355,
    #                                            self.res.params['y_stddev'].value * 2.355,
    #                                            self.res.params['theta'].value])
    #
    #     self._spotarr = np.asarray(np.sqrt(
    #                                 np.array(self._spotdict.values())[:, 3]*
    #                                 np.array(self._spotdict.values())[:, 4])
    #                                ).flatten()

    # def fitGaussian2D(self, data, err):
    #     from argos.util.image_models import ImageModels
    #
    #     def gaussian2D_deriv():
    #         pass
    #
    #     def callGaussian(p):
    #         amplitude= p['amplitude'].value
    #         x_mean= p['x_mean'].value
    #         y_mean= p['y_mean'].value
    #         x_stddev= p['x_stddev'].value
    #         y_stddev= p['y_stddev'].value
    #         theta= p['theta'].value
    #         return lambda x, y: ImageModels.gaussian2D(x, y, amplitude,
    #                                        x_mean, y_mean,
    #                                        x_stddev, y_stddev, theta)
    #
    #
    #     errorfunction = lambda par: np.ravel(np.abs(callGaussian(par)
    #                                    (*np.indices(data.shape))-
    #                                    data) / err)
    #     ret= minimize(errorfunction, self.par)
    #     return ret

    # def spotsSecondMomentCovarianceMatrixPerPupil(self):
    #     if self._momentdict is None:
    #         self._compute_image_moment()
    #     datadict= self.spotsImageMomentDictionaryPerPupil()
    #     blue= self._compute_average_covariance_matrix_per_pupil(
    #         datadict[LaserBeam.Blue])
    #     red= self._compute_average_covariance_matrix_per_pupil(
    #         datadict[LaserBeam.Red])
    #     yellow= self._compute_average_covariance_matrix_per_pupil(
    #         datadict[LaserBeam.Yellow])
    #     return {LaserBeam.Blue: blue,
    #             LaserBeam.Yellow: yellow,
    #             LaserBeam.Red: red}
