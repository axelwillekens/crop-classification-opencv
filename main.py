import cv2
import Indexer
import Thresholder
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    path = "../images/test/mais.jpg"
    img = cv2.imread(path)

    indexer = Indexer.Indexer()
    thresholder = Thresholder.Thresholder()

    # Show indexing
    indeximages = [indexer.index("ExG", img), indexer.index("ExGR", img), indexer.index("NDI", img),
                   indexer.index("CIVE", img), indexer.index("VEG", img), indexer.index("COM", img),
                   indexer.index("GA", img), indexer.index("HIT", img), img]
    indextitles = ["ExG", "ExGR", "NDI",
                   "CIVE", "VEG", "COM",
                   "GA", "HIT", "Original"]

    # for i in range(0, 3):
    #     plt.subplot(3, 3, i*3 + 1), plt.imshow(indeximages[i*3])
    #     plt.title(indextitles[i * 3]), plt.xticks([]), plt.yticks([])
    #     plt.subplot(3, 3, i*3 + 2), plt.imshow(indeximages[i*3 + 1])
    #     plt.title(indextitles[i*3 + 1]), plt.xticks([]), plt.yticks([])
    #     plt.subplot(3, 3, i*3 + 3), plt.imshow(indeximages[i*3 + 2])
    #     plt.title(indextitles[i*3 + 2]), plt.xticks([]), plt.yticks([])
    # plt.show()

    # Show Thresholding
    idxmethod = "ExGR"
    thmethod = "Kapur"
    imgidx = indexer.index(idxmethod, img)
    th, imgth = thresholder.threshold(thmethod, imgidx)
    thtitles = ["%s Threshold method" % thmethod, "Histogram, threshold on %d" % th]

    plt.subplot(1, 2, 1), plt.imshow(imgth)
    plt.title(thtitles[0]), plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2), plt.hist(imgidx.ravel(), 256)
    plt.title(thtitles[1]), plt.xticks(range(0, 256, 50)), plt.yticks(range(0, 5000, 1000)), \
        plt.xlabel("Pixel Value"), plt.ylabel("Number of pixels"), plt.axvline(th)
    plt.show()

    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
