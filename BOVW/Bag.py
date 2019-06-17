import cv2
import numpy as np
from glob import glob
import argparse
from helpers import *
from matplotlib import pyplot as plt
import json
import csv

# removed: images\train\photo\biograd-C_01674_left.jpg

runcount = '3'


class BOV:
    def __init__(self, no_clusters):
        self.no_clusters = no_clusters
        self.train_path = None
        self.test_path = None
        self.im_helper = ImageHelpers()
        self.bov_helper = BOVHelpers(no_clusters)
        self.file_helper = FileHelpers()
        self.images = None
        self.trainImageCount = 0
        self.train_labels = np.array([])
        self.name_dict = {}
        self.descriptor_list = []
        self.filenamesss = []
        self.retrain = False

    def trainModel(self):
        """
        This method contains the entire module
        required for training the bag of visual words model

        Use of helper functions will be extensive.

        """

        removed = 0
        # read file. prepare file lists.

        self.images, self.trainImageCount, self.filenamesss = self.file_helper.getFiles(self.train_path)
        # extract SIFT Features from each image
        label_count = 0
        # MINE: for word, imlist in self.images.iteritems():
        itr = 0

        for word, imlist in self.images.items():
            self.name_dict[str(label_count)] = word
            if self.retrain:
                print("Computing Features for ", word)
                # MINE
                for im in imlist:
                    # cv2.imshow("im", im)
                    # cv2.waitKey()
                    kp, des = self.im_helper.features(im)
                    # if des is not None:
                    if des is None:
                        removed += 1
                        print(self.filenamesss[itr])
                    else:
                        self.descriptor_list.append(des)
                        self.train_labels = np.append(self.train_labels, label_count)
                    itr += 1
            label_count += 1

        # perform clustering
        bov_descriptor_stack = self.bov_helper.formatND(self.descriptor_list)
        print("Obtaining Clusters")
        self.bov_helper.cluster()
        print("Developing Vocabulary")
        self.bov_helper.developVocabulary(n_images=self.trainImageCount - removed, descriptor_list=self.descriptor_list)

        # show vocabulary trained
        # self.bov_helper.plotHist()

        print("Standardizing")
        self.bov_helper.standardize()
        print("Training SVM")
        self.bov_helper.train(self.train_labels)

    def recognize(self, test_img, test_image_path=None):
        """
        This method recognizes a single image
        It can be utilized individually as well.


        """

        kp, des = self.im_helper.features(test_img)

        if des is not None:
            proceed = 1

            # generate vocab for test image
            vocab = np.array([[0 for i in range(self.no_clusters)]])
            # locate nearest clusters for each of
            # the visual word (feature) present in the image

            # test_ret =<> return of kmeans nearest clusters for N features
            test_ret = self.bov_helper.kmeans_obj.predict(des)
            # print test_ret

            # print vocab
            for each in test_ret:
                vocab[0][each] += 1

            # Scale the features
            vocab = self.bov_helper.scale.transform(vocab)

            # predict the class of the image
            lb = self.bov_helper.clf.predict(vocab)
            # print "Image belongs to class : ", self.name_dict[str(int(lb[0]))]
        else:
            proceed = 0
            lb = 0
        return lb, proceed

    def testModel(self):
        """
        This method is to test the trained classifier

        read all images from testing path
        use BOVHelpers.predict() function to obtain classes of each image

        """

        self.testImages, self.testImageCount, self.testfilenamesss = self.file_helper.getFiles(self.test_path)

        predictions = []
        ctr = 0.0
        per_class_ctr = 0.0
        correct = 0.0
        per_class_correct = 0.0
        per_class_accuracy = []
        classes = []
        # MINE: for word, imlist in self.testImages.iteritems():
        for word, imlist in self.testImages.items():
            print("processing ", word)
            classes.append(word)
            per_class_ctr = 0.0
            per_class_correct = 0.0
            for im in imlist:
                # print imlist[0].shape, imlist[1].shape
                print(im.shape)
                cl, proceed = self.recognize(im)
                if proceed == 1:
                    ctr += 1.0
                    per_class_ctr += 1.0
                    print(cl)
                    predictions.append([cl, self.name_dict[str(int(cl[0]))], word])
                    if self.name_dict[str(int(cl[0]))] == word:
                        correct += 1.0
                        per_class_correct += 1.0
            cls_acc = per_class_correct / per_class_ctr
            per_class_accuracy.append(cls_acc)

        accuracy = correct / ctr

        print(predictions)
        '''for each in predictions:
            # cv2.imshow(each['object_name'], each['image'])
            # cv2.waitKey()
            # cv2.destroyWindow(each['object_name'])
            #
            plt.imshow(cv2.cvtColor(each['image'], cv2.COLOR_GRAY2RGB))
            plt.title(each['object_name'])
            plt.show()'''

        return predictions, accuracy, per_class_accuracy, classes

    def print_vars(self):
        pass


if __name__ == '__main__':

    # parse cmd args
    '''parser = argparse.ArgumentParser(
        description=" Bag of visual words example"
    )
    parser.add_argument('--train_path', action="store", dest="train_path", required=True)
    parser.add_argument('--test_path', action="store", dest="test_path", required=True)

    args = vars(parser.parse_args())
    print(args)'''

    bov = BOV(no_clusters=100)

    # set training paths
    bov.train_path = 'images\\train\\'  # args['train_path']
    # set testing paths
    bov.test_path = 'images\\test\\'  # args['test_path']
    # train the model
    #print('training the model')
    bov.trainModel()
    # test model
    print('testing the model')
    # read saved model
    predictions, accuracy, per_class_accuracy, classes = bov.testModel()
    print("test prediction accuracy:", accuracy)
    print("classes", classes)
    print("per_class_accuracy", per_class_accuracy)

    with open('prediction_results_%s.csv' % runcount, 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerows(predictions)
