import csv
import os
import pandas as pd
import json
import numpy as np
import cv2
import shutil
from sklearn import svm
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from config import *

classes = ['none', 'start', 'end', 'up', 'down', 'photo', 'backward', 'carry', 'boat', 'here', 'mosaic', 'delimiter', 'one', 'two', 'three', 'four', 'five']
class_indices = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]


def read_csv(file, header=True):
    data = pd.read_csv(file, header=header).values.tolist()
    return data


dataset = read_csv(src_csv)


def read_data(data, labels):
    dataset = np.array(pd.read_csv(data, header=None))
    label = np.array(pd.read_csv(labels, header=None))
    label = label.reshape(-1)
    return dataset, label


def set_filelist(setloc):
    setlist = []
    folders = os.listdir(setloc)
    for folder in folders:
        if folder != 'none':
            files = os.listdir(setloc + folder)
            for file in files:
                setlist.append(file)
    return setlist


def set_filelist_none(setloc):
    setlist = []
    folders = os.listdir(setloc)
    for folder in folders:
        if folder == 'none':
            files = os.listdir(setloc + folder)
            for file in files:
                setlist.append(file)
    return setlist


train_files = set_filelist(train_images)  # returns filenames of positive samples
test_files = set_filelist(test_images)  # returns filenames of positive samples

train_files_none = set_filelist_none(train_images)
test_files_none = set_filelist_none(test_images)

# trimmed CSV data (filename, label, and ROI)
trimmed_data = []

width_list = []
height_list = []

train_list = []
test_list = []

train_list_none = []
test_list_none = []

train_right_list = []

# preparing patches from right images
'''src_img = '/home/ygi/Desktop/UP/CS282/CADDY_TRUEPOSITIVES/'
train_right_positive_patches = '/home/ygi/Desktop/UP/CS282/HOG/images/train_right_positive/'
dest_img_whole = '/home/ygi/Desktop/UP/CS282/HOG/images/train_right_positive_whole/'

for file in train_files:
    file = file.rsplit('_left')[0]
    file = file + '_right.jpg'
    exists = os.path.isfile(src_img + file)

    findme = 'true_positives/raw/' + file
    # obtain class in image
    entry = [l for l in dataset if findme in l[3]]
    class_label = entry[0][5] + 1
    class_name = classes[class_label]
    assert len(entry) == 1


    if not os.path.exists(dest_img_whole + class_name):
        os.mkdir(dest_img_whole + class_name)

    if exists:
        print(file)
        # save copy of entire image to dest_img_whole
        shutil.copy(src_img + file, dest_img_whole + str(classes[class_label]) + '/' + file)
        # obtain ROI patch from said image

        if not (class_name == 'none' or class_name == 'mosaic'):
            if entry[0][7] == entry[0][7]: # if there is an ROI annotation
                print(class_name)
                roi = json.loads(entry[0][7]) # convert ROI entry to list
                print(roi)
                roi_x = roi[0]
                roi_y = roi[1]
                roi_width = roi[2]
                roi_height = roi[3]

                # save image
                img = cv2.imread(dest_img_whole + str(classes[class_label]) + '/' + file)
                patch = img[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width, :]
                revised_filename = file.rsplit('.')[0] + class_name + '.jpg'
                train_right_list.append([revised_filename, class_label])
                # simple resizing
                resized_patch = cv2.resize(patch,(64, 80), interpolation = cv2.INTER_AREA)
                cv2.imwrite(train_right_positive_patches + revised_filename, resized_patch)

with open(train_positive_right_list, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(train_right_list)
'''

'''
# positive patches
for i, entry in enumerate(dataset):
    if 'true_positives/raw' in entry[2]:
        filename = entry[2].rsplit('/')[-1]
        print(filename)
        class_label = entry[5] + 1
        class_name = classes[class_label]
        print(entry)
        if not (class_name == 'none' or class_name == 'mosaic'):
            if entry[6] == entry[6]: # skip entries with no ROI info ('nan')
                print(class_name)
                roi = json.loads(entry[6]) # convert ROI entry to list
                print(roi)
                roi_x = roi[0]
                roi_y = roi[1]
                roi_width = roi[2]
                roi_height = roi[3]
                width_list.append(roi_width)
                height_list.append(roi_height)

            # save positive patches (original scale)
            if filename in train_files:
                assert filename not in test_files
                img = cv2.imread(train_images + class_name + "/" + filename)
                patch = img[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width, :]
                revised_filename = filename.rsplit('.')[0] + class_name + '.jpg'
                train_list.append([revised_filename, class_label])
                # simple resizing
                resized_patch = cv2.resize(patch,(64, 80), interpolation = cv2.INTER_AREA)
                cv2.imwrite(train_positive_patches + revised_filename, resized_patch)

            elif filename in test_files:
                assert filename not in train_files

                img = cv2.imread(test_images + class_name + "/" + filename)
                patch = img[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width, :]
                revised_filename = filename.rsplit('.')[0] + class_name + '.jpg'
                test_list.append([revised_filename, class_label])
                # simple resizing
                resized_patch = cv2.resize(patch,(64, 80), interpolation = cv2.INTER_AREA)
                cv2.imwrite(test_positive_patches + revised_filename, resized_patch)
'''

# negative patches
'''
for i, entry in enumerate(dataset):
    if 'true_negatives/raw' in entry[2]:
        filename = entry[2].rsplit('/')[-1]
        print(filename)
        class_label = entry[5] + 1
        class_name = classes[class_label]
        print(entry)
        x_random = np.random.randint(0, 512)
        y_random = np.random.randint(0, 320)

        # save negative_patches
        if filename in train_files_none:
            assert filename not in test_files_none
            img = cv2.imread(train_images + class_name + "/" + filename)
            patch = img[y_random:y_random+160, x_random:x_random+128, :]
            revised_filename = filename.rsplit('.')[0] + class_name + '.jpg'
            train_list_none.append([revised_filename, class_label])
            resized_patch = cv2.resize(patch,(64, 80), interpolation = cv2.INTER_AREA)
            cv2.imwrite(train_negative_whole + revised_filename, img)
            #cv2.imwrite(train_negative_patches + revised_filename, resized_patch)

        elif filename in test_files_none:
            assert filename not in train_files_none
            img = cv2.imread(test_images + class_name + "/" + filename)
            patch = img[y_random:y_random+160, x_random:x_random+128, :]
            revised_filename = filename.rsplit('.')[0] + class_name + '.jpg'
            test_list_none.append([revised_filename, class_label])
            resized_patch = cv2.resize(patch,(64, 80), interpolation = cv2.INTER_AREA)
            cv2.imwrite(test_negative_whole + revised_filename, img)
            #cv2.imwrite(test_negative_patches + revised_filename, resized_patch)'''


# img height = 480
# img width = 640
mean_width = np.mean(width_list)  # ----> 121.79, max: 376
mean_height = np.mean(height_list)  # ----> 149.24, max: 391
stdev_width = np.std(width_list)  # ----> 47.90
stdev_height = np.std(height_list)  # ----> 50.49
ratio = np.divide(width_list, height_list)  # ----> mean width:height ratio: 0.82

'''with open(train_positive_list, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(train_list)

with open(test_positive_list, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(test_list)

with open(train_negative_list, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(train_list_none)

with open(test_negative_list, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(test_list_none)'''

# HOG
winSize = (64, 80)
cellSize = (8, 8)
blockSize = (16, 16)
blockStride = (8, 8)
nbins = 9
derivAperture = 1
winSigma = -1
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = 1
nlevels = 64
useSignedGradients = False
c_val = 0.1
kernel = 'linear'
include_negative = True
include_right = True
include_hnm = True

train_dataset = read_csv(train_positive_list, header=None)
if include_negative:
    train_dataset = train_dataset + read_csv(train_negative_list, header=None)
else:
    print('Excluding Negative Samples')
if include_right:
    train_dataset = train_dataset + read_csv(train_positive_right_list, header=None)
else:
    print('Excluding Right Samples')
if include_hnm:
    train_dataset = train_dataset + read_csv(train_difficult_patches_list, header=None)
else:
    print('Excluding HNM Samples')

test_dataset = read_csv(test_positive_list, header=None)
if include_negative:
    test_dataset = test_dataset + read_csv(test_negative_list, header=None)


def compute_hog(data_set, images_set, y_csv, x_csv):
    descriptors = []
    labels = []
    for i, img in enumerate(data_set):
        print(img)
        labels.append(img[1])
        img = cv2.imread(images_set + img[0])
        hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma, histogramNormType, L2HysThreshold, gammaCorrection, nlevels, useSignedGradients)
        descriptor = hog.compute(img)
        descriptors.append(np.array(descriptor).transpose()[0])

    np.savetxt(y_csv, labels)
    with open(x_csv, 'w', newline='') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerows(descriptors)


compute_hog(train_dataset, train_dataset_images, y_train_csv, x_train_csv)
'''compute_hog(test_dataset, test_dataset_images, y_test_csv, x_test_csv)'''


# Training
x_train, y_train = read_data(x_train_csv, y_train_csv)
#x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.0001)
print('Fitting the SVM Model')
clf = svm.SVC(C=c_val, kernel=kernel, cache_size=1000, probability=True)  # decision_function_shape = ovr
#clf.fit(x_train, y_train)
#dump(clf, models + 'hog_svm_%s_%s_withhnm_right_none.joblib' % (c_val, kernel))
clf = load(models + 'hog_svm.joblib')
decision_function = clf.decision_function(x_train)

'''
# HARD NEGATIVE MINING
# Sliding Window over Negative IMAGES
# Extract patches that have been incorrectly classified as positive patches. Take also their prediction confidence
sliding_window_stride = 16
negative_samples = read_csv(train_negative_list, header=None)
negative_descriptors = []
negative_labels = []
negative_savedpatches = []
negative_decision_scores = []
hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma, histogramNormType, L2HysThreshold, gammaCorrection, nlevels, useSignedGradients)
x = 0
y = 0
count = 0
for i, neg in enumerate(negative_samples):
    print('image #', i, neg[0])
    name_of_file = neg[0]
    neg = cv2.imread(train_negative_whole + neg[0])
    neg_small = cv2.resize(neg, (400, 300), interpolation = cv2.INTER_AREA)
    neg_large = cv2.resize(neg, (1024, 768), interpolation = cv2.INTER_AREA)

    # normal scale
    y = 0
    while y < 768 - 160:
        x = 0
        while x < 1024 - 128 :
            patches = []
            patch_names = []
            # patch_normal (640, 480)
            if y <= 480 - 160 and x <= 640 - 128:
                patch_normal = neg[y:y+160, x:x+128, :]
                patch_normal = cv2.resize(patch_normal, (64, 80), interpolation = cv2.INTER_AREA)
                patches.append(patch_normal)
                patch_names.append('normal')
            # patch_small
            if y <= 300 - 160 and x <= 400 - 128:
                patch_small = neg_small[y:y+160, x:x+128, :]
                patch_small = cv2.resize(patch_small, (64, 80), interpolation = cv2.INTER_AREA)
                patches.append(patch_small)
                patch_names.append('small')
            # patch_large
            if y <= 768 - 160 and x <= 1024 - 128:
                patch_large = neg_large[y:y+160, x:x+128, :]
                patch_large = cv2.resize(patch_large, (64, 80), interpolation = cv2.INTER_AREA)
                patches.append(patch_large)
                patch_names.append('large')

            for k, patch in enumerate(patches):
                descriptor = hog.compute(patch)     # compute HOG descriptor for negative patch
                assert descriptor.shape == (2268, 1)
                # Run through SVM Model
                prediction_on_negative = clf.predict(descriptor.transpose())

                # If false positive, save difficult to classify patches
                if prediction_on_negative != 0:
                    print("DIFFICULT", name_of_file, 'at', x, ',', y)
                    decision_function = clf.decision_function(descriptor.transpose())
                    negative_decision_scores.append(decision_function[0])
                    negative_descriptors.append(np.array(descriptor).transpose()[0])
                    negative_labels.append(0)
                    negative_savedpatches.append(patch_names[k] + '_' + str(x) + '_' + str(y) + '_' + name_of_file)
                    cv2.imwrite(difficult_patches + patch_names[k] + '_' + str(x) + '_' + str(y) + '_' + name_of_file, patch)
                else:
                    count += 1

            x += sliding_window_stride
        y += sliding_window_stride

# hard negative mining
hnm_decision_scores = '/home/ygi/Desktop/UP/CS282/HOG/images/hnm_decision_scores.csv'
hnm_descriptors = '/home/ygi/Desktop/UP/CS282/HOG/images/hnm_descriptors.csv'
hnm_patchnames = '/home/ygi/Desktop/UP/CS282/HOG/images/hnm_patchnames.csv'

with open(hnm_decision_scores, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(negative_decision_scores)

with open(hnm_descriptors, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(negative_descriptors)

with open(hnm_patchnames, "w", newline="") as f:
    writer = csv.writer(f)
    for neg_patch in negative_savedpatches:
        writer.writerow([neg_patch, 0])
'''
