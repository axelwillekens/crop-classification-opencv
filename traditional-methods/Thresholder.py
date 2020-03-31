import cv2
import numpy as np


class Thresholder:
    def __init__(self):
        self.thtable = {"otsu": self.thresholdotsu, "ridler": self.thresholdridler, "kapur": self.thresholdkapur,
                               "kittler": self.thresholdkittler, "rosin": self.thresholdrosin}
        self.idxtable = {"exg": cv2.THRESH_BINARY_INV, "exgr": cv2.THRESH_BINARY_INV, "ndi": cv2.THRESH_BINARY_INV,
                         "cive": cv2.THRESH_BINARY,  "veg": cv2.THRESH_BINARY_INV, "com": cv2.THRESH_BINARY_INV,
                         "ga": cv2.THRESH_BINARY_INV, "hit": cv2.THRESH_BINARY}

    def threshold(self, thtype, idxtype, img):
        """
        Arguments:
        Image has to be a gray value image
        Supported indexes: Otsu, Ridler, Kapur, Kittler, Rosin
        Returns:
            Tuple (threshold value, binary image)
        """
        idxtype = str(idxtype).lower()
        assert idxtype in self.idxtable.keys(), "Assert: Index does not exist."

        thtype = str(thtype).lower()
        assert thtype in self.thtable.keys(), "Assert: Threshold does not exist."

        return self.thtable[thtype](img, idxtype)

    def thresholdotsu(self, img, idxtype):
        th, imgth = cv2.threshold(img, 0, 255, self.idxtable[idxtype]+cv2.THRESH_OTSU)
        return th, imgth

    def thresholdridler(self, img, idxtype):
        imgflat = img.ravel()
        th_old = np.mean(imgflat)
        th_new = 0
        diff = abs(th_old - th_new)
        hist, bin_edges = np.histogram(img.ravel(), 255)
        intensities = hist * bin_edges[:-1]

        while diff > 1:
            th_new = (sum(intensities[:int(th_old)])/sum(hist[:int(th_old)]) + sum(intensities[int(th_old):])/sum(hist[int(th_old):])) / 2
            diff = abs(th_old - th_new)
            th_old = th_new

        th, imgth = cv2.threshold(img, th_new, 255, self.idxtable[idxtype])
        return th, imgth

    def thresholdkapur(self, img, idxtype):
        imgflat = img.ravel()
        hist, bin_edges = np.histogram(imgflat, 255)
        # if 0: H(X) = - SUM( P(X_g) * log(P(X_g))) -- is 0 * -inf = NaN => incr with 0.001 to prevent this
        nonzerohist = hist + 0.001

        Hmax = - float("inf")
        th = 0
        for i in range(1, 255):
            Pf = nonzerohist[:i].sum()
            Pb = nonzerohist[i:].sum()
            Hf = -(np.divide(nonzerohist[:i], Pf)*np.log(np.abs(np.divide(nonzerohist[:i], Pf)))).sum()
            Hb = -(np.divide(nonzerohist[i:], Pb)*np.log(np.abs(np.divide(nonzerohist[i:], Pb)))).sum()
            Hsum = Hf + Hb
            if Hsum > Hmax:
                Hmax = Hsum
                th = i

        th, imgth = cv2.threshold(img, th, 255, self.idxtable[idxtype])
        return th, imgth

    def thresholdkittler(self, img, idxtype):
        # How to calculate , histogram is not complete
        ret = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return ret

    def thresholdrosin(self, img, idxtype):
        # How to implement, histogram is not complete
        ret = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return ret