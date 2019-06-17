# CLASSIFICATION ON THE ENTIRE IMAGE
import os
import cv2
import csv
from joblib import dump, load
import pandas as pd
import numpy as np
import random
from config import test_images, models, results


random.seed(6)

#classes = ['none', 'start', 'end', 'up', 'down', 'photo', 'backward', 'carry', 'boat', 'here', 'mosaic', 'delimiter', 'one', 'two', 'three', 'four', 'five']
#class_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
classes = ['none', 'start', 'end', 'up', 'down', 'photo', 'backward', 'carry', 'boat', 'here', 'delimiter', 'one', 'two', 'three', 'four', 'five']
class_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

folders = os.listdir(test_images)

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

sliding_window_stride = 16

clf = load(models + 'hog_svm.joblib' % (c_val, kernel))
hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma, histogramNormType, L2HysThreshold, gammaCorrection, nlevels, useSignedGradients)

for folder in folders:
    print("testing class", folder)
    if folder == 'mosaic':
        print('skipping mosaic')
        continue
    with open(results + folder + '.csv', 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['filename', 'target', 'prediction', 'bboxes'])
    to_save = []
    target_label = folder
    target_class = classes.index(target_label)
    test_imgs = os.listdir(test_images + folder)
    random.shuffle(test_imgs)

    for kk, test_img in enumerate(test_imgs):
        if kk >= 80:
            with open(results + folder + '.csv', 'a', newline='') as f:
                wr = csv.writer(f)
                wr.writerows(to_save)
            break
        my_file = test_img
        print(kk, test_img)
        test_img = cv2.imread(test_images + folder + '/' + test_img)
        x = 0
        y = 0
        count = 0
        bboxes = []
        prediction_scores = []
        prediction_class = []

        # normal scale
        y = 0
        while y < 480 - 100:
            x = 0
            while x < 640 - 80:
                patches = []
                patch_names = []
                bboxes_candidates = []
                # patch_normal (640, 480)
                if y <= 480 - 160 and x <= 640 - 128:
                    patch_normal = test_img[y:y + 160, x:x + 128, :]
                    patch_normal = cv2.resize(patch_normal, (64, 80), interpolation=cv2.INTER_AREA)
                    patches.append(patch_normal)
                    patch_names.append('normal')
                    bboxes_candidates.append([x, y, x + 128, y + 160])
                # patch_small
                if y <= 480 - 256 and x <= 640 - 205:
                    patch_small = test_img[y:y + 256, x:x + 205, :]
                    patch_small = cv2.resize(patch_small, (64, 80), interpolation=cv2.INTER_AREA)
                    patches.append(patch_small)
                    patch_names.append('small')
                    bboxes_candidates.append([x, y, x + 205, y + 256])
                # patch_large
                if y <= 480 - 100 and x <= 640 - 80:
                    patch_large = test_img[y:y + 100, x:x + 80, :]
                    patch_large = cv2.resize(patch_large, (64, 80), interpolation=cv2.INTER_AREA)
                    patches.append(patch_large)
                    patch_names.append('large')
                    bboxes_candidates.append([x, y, x + 80, y + 100])

                for k, patch in enumerate(patches):
                    descriptor = hog.compute(patch)     # compute HOG descriptor for negative patch
                    assert descriptor.shape == (2268, 1)
                    # Run through SVM Model
                    prediction = clf.predict_proba(descriptor.transpose())
                    prediction = prediction[0]
                    assert prediction.shape[0] == 16  # exluding mosaic
                    prediction_max_index = np.argmax(prediction)
                    prediction_max = np.max(prediction)

                    # If prediction is not 'none':
                    if prediction_max_index != 0:
                        prediction_scores.append(prediction_max)
                        prediction_class.append(prediction_max_index)
                        bboxes.append(bboxes_candidates[k])

                x += sliding_window_stride
            y += sliding_window_stride

        # choosing the max class
        if len(bboxes) == 0:
            final_prediction = 0
            chosen_bbox = 'empty'
        else:
            assert len(prediction_scores) == len(prediction_class)
            max_pred_index = np.argmax(prediction_scores)
            final_prediction = prediction_class[max_pred_index]
            chosen_bbox = bboxes[max_pred_index]

        to_save.append([my_file, target_class, final_prediction, chosen_bbox])
        print('target class:', target_class, 'predicted class:', final_prediction)

    with open(results + folder + '.csv', 'a', newline='') as f:
        wr = csv.writer(f)
        wr.writerows(to_save)
