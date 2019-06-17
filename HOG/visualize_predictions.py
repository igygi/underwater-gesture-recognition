# Visualization Test Detection Results

import os
import pandas as pd
import cv2
import numpy as np
import random
import json
from config import test_images, results, visualization_results

random.seed(6)

orig_classes = ['none', 'start', 'end', 'up', 'down', 'photo', 'backward', 'carry', 'boat', 'here', 'delimiter', 'one', 'two', 'three', 'four', 'five']
classes = ['start', 'end', 'up', 'down', 'photo', 'backward', 'carry', 'boat', 'here', 'delimiter', 'one', 'two', 'three', 'four', 'five']


print('Visualizing. Taking at most 5 correct and incorrect samples for each class. Note: excluded none class')

num_correct_vis = 5
num_incorrect_vis = 5

font = cv2.FONT_HERSHEY_SIMPLEX


def read_csv(file, header=None):
    data = pd.read_csv(file, header=header).values.tolist()
    return data


for gesture in classes:
    correct_vis = 0
    incorrect_vis = 0
    predictions = read_csv(results + gesture + '.csv', header=['filename', 'target', 'prediction', 'bboxes'])
    random.shuffle(predictions)
    for entry in predictions:
        x1 = 0
        y1 = 0
        x2 = 0
        y2 = 0
        if incorrect_vis >= 5 and correct_vis >= 5:
            break
        if entry[1] == entry[2] and gesture != 'none' and correct_vis < 5 and entry[2] != 0:
            img = entry[0]
            img = cv2.imread(test_images + gesture + '/' + img)
            correct_vis += 1
            bbox = entry[3]
            bbox = bbox.replace('[', '')
            bbox = bbox.replace(']', '')
            bbox = bbox.replace(' ', '')
            bbox = bbox.split(',')
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2])
            y2 = int(bbox[3])

            # draw bbox on image
            img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            img = cv2.putText(img, orig_classes[entry[2]], (10, 50), font, 2.0, (0, 255, 0), 2, cv2.LINE_8)

            # save image
            savename = visualization_results + gesture + '_correct_' + entry[0]
            print(savename)
            cv2.imwrite(savename, img)

        elif entry[1] != entry[2] and gesture != 'none' and incorrect_vis < 5 and entry[2] != 0:
            img = entry[0]
            img = cv2.imread(test_images + gesture + '/' + img)
            incorrect_vis += 1
            bbox = entry[3]
            bbox = bbox.replace('[', '')
            bbox = bbox.replace(']', '')
            bbox = bbox.replace(' ', '')
            bbox = bbox.split(',')
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2])
            y2 = int(bbox[3])

            # draw bbox on image
            img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            img = cv2.putText(img, orig_classes[entry[2]], (10, 50), font, 2.0, (0, 0, 255), 2, cv2.LINE_8)

            # save image
            savename = visualization_results + gesture + '_incorrect_' + entry[0]
            print(savename)
            cv2.imwrite(savename, img)
