import cv2
import Indexer
import Thresholder
import FeatureExtractor
import numpy as np
from matplotlib import pyplot as plt
import os
import pylab as pl
from scipy.cluster.vq import kmeans, vq
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib


def imglist(path):
    return [os.path.join(path, f) for f in os.listdir(path)]


def showconfusionmatrix(cm):
    pl.matshow(cm)
    pl.title("Confusion matrix")
    pl.colorbar()
    pl.show()


indexer = Indexer.Indexer()
thresholder = Thresholder.Thresholder()
featureExtractor = FeatureExtractor.FeatureExtractor()

idxmethod = "COM"
thmethod = "Ridler"
ftype = "SIFT"

if __name__ == "__main__":
    # loat the classifier, class names, scaler, number of clusters and vocabulary
    # from stored pickle file (generated during training)
    clf, classes_names, stdSlr, k, voc = joblib.load("bovw.pkl")

    # Get the path of the testing image(s) and store them in a list
    test_path = '../dataset/test'
    testing_names = os.listdir(test_path)

    # Get path to all images and save them in a list
    # img_paths and the corresponding label in img_paths
    img_paths = []
    img_classes = []
    class_id = 0

    for testing_name in testing_names:
        dir_name = os.path.join(test_path, testing_name)
        class_path = imglist(dir_name)
        img_paths += class_path
        img_classes += [class_id]*len(class_path)
        class_id += 1

    # Create feature extraction and keypoint detector objects
    # Create list where all the descriptors will be stored
    des_list = []

    for img_path in img_paths:
        img = cv2.imread(img_path)
        imgidx = indexer.index(idxmethod, img)
        th, imgth = thresholder.threshold(thmethod, idxmethod, imgidx)
        kp, des = featureExtractor.feature(ftype, img, imgth)
        des_list.append((img_path, des))

    # stack all the descriptors vertically in a numpy array
    descriptors = des_list[0][1]
    for img_path, descriptor in des_list[1:]:
        descriptors = np.vstack((descriptors, descriptor))

    # Calculate the histogram of features
    # vq Assigns codes from a code book to observations
    im_features = np.zeros((len(img_paths), k), "float32")
    for i in range(len(img_paths)):
        words, distance = vq(des_list[i][1], voc)
        for w in words:
            im_features[i][w] += 1

    # Perform If-Idf vectorization
    nbr_occurences = np.sum((im_features > 0) * 1, axis=0)
    idf = np.array(np.log((1.0 * len(img_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

    # Scale the features
    # Standardize features by removing the mean and scaling to unit variance
    # Scaler (stdSlr comes from the pickled file we imported)
    im_features = stdSlr.transform(im_features)

    # Report true class names so they can be compared with predicted classes
    true_class = [classes_names[i] for i in img_classes]
    # Perform the predictions and report predicted class names.
    predictions = [classes_names[i] for i in clf.predict(im_features)]

    # Print the true class and Predictions
    print("true_class = " + str(true_class))
    print("prediction = " + str(predictions))

    ####################################################
    # To make it easy to understand the accuracy let us print the confusion matrix
    accuracy = accuracy_score(true_class, predictions)
    print("accuracy = ", accuracy)
    cm = confusion_matrix(true_class, predictions)
    print(cm)

    showconfusionmatrix(cm)



