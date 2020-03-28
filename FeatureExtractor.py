import cv2


class FeatureExtractor:
    def __init__(self):
        self.featureTable = {"sift": self.featureSIFT, "surf": self.featureSURF}

    def feature(self, featuretype, img, mask):
        """
        Image has to be in BGR color shape, mask has to be added
        Supported indexes: SIFT, SURF
        """
        featuretype = str(featuretype).lower()
        if featuretype not in self.featureTable.keys():
            print("ERROR: Feature does not exist -- None returned")
            return None
        else:
            return self.featureTable[featuretype](img, mask)

    def featureSIFT(self, img, mask):
        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(img, mask)
        return kp, des

    def featureSURF(self, img, mask):
        surf = cv2.xfeatures2d.SURF_create()
        kp, des = surf.detectAndCompute(img, mask)
        return kp, des

