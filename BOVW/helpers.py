import cv2
import numpy as np
from glob import glob
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import pickle

runcount = '3'


class ImageHelpers:
    def __init__(self):
        self.sift_object = cv2.xfeatures2d.SIFT_create()
        #self.sift_object = cv2.ORB_create()

    def gray(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray

    def features(self, image):
        #keypoints, descriptors = self.sift_object.detectAndCompute(image, None)
        keypoints, descriptors = self.sift_object.detectAndCompute(image, None)
        return [keypoints, descriptors]


class BOVHelpers:
    def __init__(self, n_clusters=20):
        self.n_clusters = n_clusters
        self.kmeans_obj = KMeans(n_clusters=n_clusters)
        self.kmeans_ret = None
        self.descriptor_vstack = None
        self.mega_histogram = None
        self.clf = SVC()
        self.retrain = False  # True #False
        self.retrain_cluster = False  # True #False
        self.retrain_svm = False  # True #False

    def cluster(self):
        """
        cluster using KMeans algorithm,

        """
        if self.retrain_cluster:
            print("CLUSTERING USING KMEANS ALGORITHM")
            self.kmeans_ret = self.kmeans_obj.fit_predict(self.descriptor_vstack)
            filename = 'kmeans_cluster_%s.sav' % runcount
            try:
                pickle.dump(self.kmeans_obj, open(filename, 'wb'))
            except:
                print("unable to save kmeans model")
        else:
            filename = 'kmeans_cluster_%s.sav' % runcount
            self.kmeans_obj = pickle.load(open(filename, 'rb'))
            self.kmeans_ret = self.kmeans_obj.predict(self.descriptor_vstack)
        #filename = 'kmeans_cluster.sav'
        #self.kmeans_ret = pickle.load(open(filename, 'rb'))

    def developVocabulary(self, n_images, descriptor_list, kmeans_ret=None):
        """
        Each cluster denotes a particular visual word
        Every image can be represeted as a combination of multiple
        visual words. The best method is to generate a sparse histogram
        that contains the frequency of occurence of each visual word

        Thus the vocabulary comprises of a set of histograms of encompassing
        all descriptions for all images

        """

        print("DEVELOPING VOCABULARY")
        if self.retrain:
            self.mega_histogram = np.array([np.zeros(self.n_clusters) for i in range(n_images)])
            old_count = 0
            for i in range(n_images):
                l = len(descriptor_list[i])
                for j in range(l):
                    if kmeans_ret is None:
                        idx = self.kmeans_ret[old_count + j]
                    else:
                        idx = kmeans_ret[old_count + j]
                    self.mega_histogram[i][idx] += 1
                old_count += l
            print("Vocabulary Histogram Generated")

            filename = 'mega_histogram_%s.pkl' % runcount
            try:
                pickle.dump(self.mega_histogram, open(filename, 'wb'))
            except:
                print("unable to save mega histogram model")
        else:
            filename = 'mega_histogram_%s.pkl' % runcount
            self.mega_histogram = pickle.load(open(filename, 'rb'))

    def standardize(self, std=None):
        """

        standardize is required to normalize the distribution
        wrt sample size and features. If not normalized, the classifier may become
        biased due to steep variances.

        """
        if std is None:
            self.scale = StandardScaler().fit(self.mega_histogram)
            self.mega_histogram = self.scale.transform(self.mega_histogram)
        else:
            print("STD not none. External STD supplied")
            self.mega_histogram = std.transform(self.mega_histogram)

    def formatND(self, l):
        """
        restructures list into vstack array of shape
        M samples x N features for sklearn

        """
        if self.retrain:
            vStack = np.array(l[0])
            i = 0
            for remaining in l[1:]:
                #print('REMAINING', remaining)
                i += 1
                print('Remaining', i)
                vStack = np.vstack((vStack, remaining))
            self.descriptor_vstack = vStack.copy()

            filename = 'vstack_%s.pkl' % runcount
            try:
                pickle.dump(self.descriptor_vstack, open(filename, 'wb'))
            except:
                print("unable to save vstack model")
        else:
            filename = 'vstack_%s.pkl' % runcount
            vStack = pickle.load(open(filename, 'rb'))
            self.descriptor_vstack = pickle.load(open(filename, 'rb'))

        return vStack

    def train(self, train_labels):
        """
        uses sklearn.svm.SVC classifier (SVM)


        """
        print("Training SVM")
        print(self.clf)

        if self.retrain_svm:
            print("Train labels", train_labels)
            self.clf.fit(self.mega_histogram, train_labels)
            print("Training completed")
            filename = 'svm_trained_%s.pkl' % runcount
            try:
                pickle.dump(self.clf, open(filename, 'wb'))
            except:
                print("unable to save svm model")
        else:
            filename = 'svm_trained_%s.pkl' % runcount
            self.clf = pickle.load(open(filename, 'rb'))

    def predict(self, iplist):
        predictions = self.clf.predict(iplist)
        return predictions

    def plotHist(self, vocabulary=None):
        print("Plotting histogram")
        if vocabulary is None:
            vocabulary = self.mega_histogram

        x_scalar = np.arange(self.n_clusters)
        y_scalar = np.array([abs(np.sum(vocabulary[:, h], dtype=np.int32)) for h in range(self.n_clusters)])

        print(y_scalar)

        plt.bar(x_scalar, y_scalar)
        plt.xlabel("Visual Word Index")
        plt.ylabel("Frequency")
        plt.title("Complete Vocabulary Generated")
        plt.xticks(x_scalar + 0.4, x_scalar)
        plt.show()


class FileHelpers:

    def __init__(self):
        pass

    def getFiles(self, path):
        """
        - returns  a dictionary of all files
        having key => value as  objectname => image path

        - returns total number of files.

        """
        imlist = {}
        borge = []
        count = 0
        for each in glob(path + "*"):
            print("path:", path)
            print("each:", each)
            word = each.split("\\")[-1]
            print(" #### Reading image category ", word, " ##### ")
            imlist[word] = []
            for imagefile in glob(path + word + "\*"):
                print("Reading file ", imagefile)
                im = cv2.imread(imagefile, 0)
                imlist[word].append(im)
                borge.append(imagefile)
                count += 1

        return [imlist, count, borge]
