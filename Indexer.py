import cv2
import numpy as np


class Indexer:
    def __init__(self):
        self.indextable = {"exg": self.indexExG, "exgr": self.indexExGR, "ndi": self.indexNDI, "cive": self.indexCIVE,
                           "veg": self.indexVEG, "com": self.indexCOM, "ga": self.indexGA, "hit": self.indexHIT}

    def index(self, index, img):
        """
        Image has to be in BGR color shape
        Supported indexes: ExG, ExGR, NDI, CIVE, VEG, COM, GA, HIT
        """
        index = str(index).lower()
        assert index in self.indextable.keys(), "Assert: Index does not exist."
        return self.indextable[index](img)

    def indexExG(self, img):
        ret = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        for i in range(0, img.shape[0]):
            for j in range(0, img.shape[1]):
                B = float(img[i][j][0])
                G = float(img[i][j][1])
                R = float(img[i][j][2])

                RGB = R+G+B
                r = R / RGB
                g = G / RGB
                b = B / RGB

                index = 2 * g - r - b
                ret[i][j] = index * 255

        return ret

    def indexExGR(self, img):
        ret = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        for i in range(0, img.shape[0]):
            for j in range(0, img.shape[1]):
                B = float(img[i][j][0])
                G = float(img[i][j][1])
                R = float(img[i][j][2])

                RGB = R+G+B
                r = R / RGB
                g = G / RGB
                b = B / RGB

                index = (2 * g - r - b) - (1.4 * r - g)
                ret[i][j] = index * 255

        return ret

    def indexNDI(self, img):
        ret = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        for i in range(0, img.shape[0]):
            for j in range(0, img.shape[1]):
                # B = float(img[i][j][0])
                G = float(img[i][j][1])
                R = float(img[i][j][2])

                index = (G - R) / (G + R)
                ret[i][j] = index * 255

        return ret

    def indexCIVE(self, img):
        ret = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        for i in range(0, img.shape[0]):
            for j in range(0, img.shape[1]):
                B = float(img[i][j][0])
                G = float(img[i][j][1])
                R = float(img[i][j][2])

                RGB = R+G+B
                r = R / RGB
                g = G / RGB
                b = B / RGB

                index = 0.441 * r - 0.811 * g + 0.385 * b   # + 18.78745

                ret[i][j] = index * 255

        return ret

    def indexVEG(self, img):
        ret = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        for i in range(0, img.shape[0]):
            for j in range(0, img.shape[1]):
                a = 0.667
                B = float(img[i][j][0]) + 0.001  # otherwise division by zero
                G = float(img[i][j][1])
                R = float(img[i][j][2]) + 0.001  # otherwise division by zero

                index = G / (pow(R, a) * pow(B, (1-a)))
                ret[i][j] = index * 255

        return ret

    def indexCOM(self, img):
        ret = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        for i in range(0, img.shape[0]):
            for j in range(0, img.shape[1]):
                a = 0.667
                B = float(img[i][j][0]) + 0.001  # otherwise division by zero
                G = float(img[i][j][1])
                R = float(img[i][j][2]) + 0.001  # otherwise division by zero

                RGB = R+G+B
                r = R / RGB
                g = G / RGB
                b = B / RGB

                indexexg= 2 * g - r - b
                indexcive = 0.441 * r - 0.811 * g + 0.385 * b + 18.78745
                indexveg = G / (pow(R, a) * pow(B, (1 - a)))

                index = 0.36 * indexexg + 0.47 * indexcive + 0.17 * indexveg
                ret[i][j] = index * 255

        return ret

    def indexGA(self, img):
        ret = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        for i in range(0, img.shape[0]):
            for j in range(0, img.shape[1]):
                a = 0.667
                B = float(img[i][j][0]) + 0.001  # otherwise division by zero
                G = float(img[i][j][1])
                R = float(img[i][j][2]) + 0.001  # otherwise division by zero

                RGB = R+G+B
                r = R / RGB
                g = G / RGB
                b = B / RGB

                indexexg = 2 * g - r - b
                indexcive = 0.441 * r - 0.811 * g + 0.385 * b + 18.78745
                indexveg = G / (pow(R, a) * pow(B, (1 - a)))

                index = (0.36 * indexexg + 0.47 * indexcive + 0.17 * indexveg) * g
                ret[i][j] = index * 255

        return ret

    def indexHIT(self, img):
        ret = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgxyz = cv2.cvtColor(img, cv2.COLOR_BGR2XYZ)

        for i in range(0, img.shape[0]):
            for j in range(0, img.shape[1]):
                a = 0.60
                X = float(imgxyz[i][j][0])
                Y = float(imgxyz[i][j][1])
                Z = float(imgxyz[i][j][2])

                XYZ = X + Y + Z
                x = X / XYZ
                y = Y / XYZ
                # z = Z / XYZ

                index = pow(a, (x/y))
                ret[i][j] = index * 255

        return ret
