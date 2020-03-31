import cv2
import Indexer
import Thresholder
import FeatureExtractor
import numpy as np
from matplotlib import pyplot as plt
import os

from scipy.cluster.vq import kmeans, vq
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib


def imglist(path):
    return [os.path.join(path, f) for f in os.listdir(path)]


indexer = Indexer.Indexer()
thresholder = Thresholder.Thresholder()
featureExtractor = FeatureExtractor.FeatureExtractor()

idxmethod = "COM"
thmethod = "Ridler"
ftype = "SIFT"


if __name__ == "__main__":
    train_path = '../dataset/train'
    training_names = os.listdir(train_path)

    img_paths = []
    img_classes = []
    class_id = 0

    for training_name in training_names:
        dir_name = os.path.join(train_path, training_name)
        class_path = imglist(dir_name)
        img_paths += class_path
        img_classes += [class_id]*len(class_path)
        class_id += 1

    des_list = []  # description list

    for img_path in img_paths:
        img = cv2.imread(img_path)
        imgidx = indexer.index(idxmethod, img)
        th, imgth = thresholder.threshold(thmethod, idxmethod, imgidx)
        kp, des = featureExtractor.feature(ftype, img, imgth)
        assert (des is not None), "ImagePaths %s has no features, may be too dark!" % img_path
        des_list.append((img_path, des))

    # stack all the descriptors vertically in a numpy array
    descriptors = des_list[0][1]
    for img_path, descriptor in des_list[1:]:
        descriptors = np.vstack((descriptors, descriptor))

    # kmeans works only on float thus convert intergers to float
    descriptors_float = descriptors.astype(float)

    # Perform k-means clustering and vector quantization
    k = 200  # number of clusters
    voc, variance = kmeans(descriptors_float, k, 1)

    # Calculate the histogram of features and represent them as a vector
    # vq Assigns codes from a code book to observations
    im_features = np.zeros((len(img_paths), k), "float32")
    for i in range(len(img_paths)):
        words, distance = vq(des_list[i][1], voc)
        for w in words:
            im_features[i][w] += 1

    # Perform If-Idf vectorization
    nbr_occurences = np.sum((im_features > 0) * 1, axis=0)
    idf = np.array(np.log((1.0 * len(img_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

    # Scaling the words
    # Standardize features by removing the mean and scaling to unit variance
    # In a way normalization
    stdSlr = StandardScaler().fit(im_features)
    im_features = stdSlr.transform(im_features)

    # Train an algorithm to discriminate vectors corresponding to positive and negative training
    # Train the linear SVM
    clf = LinearSVC(max_iter=50000)  # Default of 100 is not converging
    clf.fit(im_features, np.array(img_classes))

    # Train Random forest to compare how it does against SVM
    # from sklearn.ensemble import RandomForestClassifier
    # clf = RandomForestClassifier(n_estimators=100, random_state=30)
    # clf.fit(im_features, np.array(img_classes))

    # Save the SVM
    # Joblib dumps python object into one file
    joblib.dump((clf, training_names, stdSlr, k, voc), "bovw.pkl", compress=3)
