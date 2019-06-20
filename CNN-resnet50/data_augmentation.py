import os, os.path
import sys
import time
import pickle
import random
import pandas as pd
import numpy as np
import skimage
from skimage import data, io
import cv2
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

IMG_HEIGHT = 480
IMG_WIDTH = 640
IMG_CHANNELS = 3
random.seed(6)

# Local directory
# IMG_DIR = '/Volumes/NAVIVOKAJ/CS282_MP/scripts/images/'

# Online 
IMG_DIR = './images/'
DATA_DIR = "./data/"
CLASSES = ['none', 'start', 'end', 'up', 'down', 'photo', 'backward', 'carry', 'boat', 'here', 'mosaic', 'delimiter', 'one', 'two', 'three', 'four', 'five']
NUM_CLASSES = 17

def get_image_count(IMG_DIR, CLASSES, NUM_CLASSES):
    data_summary = pd.DataFrame(CLASSES)
    data_summary.columns = ['label']
    data_summary['label_id'] = range(-1,NUM_CLASSES-1)
    
    for split in ['train', 'test']:
        split_count = []
        for j in range(NUM_CLASSES):
            DIR = IMG_DIR + split + "/" + CLASSES[j]
            class_count = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
            split_count.append(class_count)
        data_summary[split] = split_count
    
    data_summary['total'] = data_summary['train'] + data_summary['test']
    
    return data_summary

def flip(img):
    # use case: for left-handed people
    new_img = np.fliplr(img)
    return new_img

def rotate(img, angle):
    new_img = skimage.transform.rotate(img, angle=angle, mode='reflect')
    return new_img

def scale(img, zoom_factor):
    height, width = img.shape[:2] # This is the final desired shape as well
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)

    ### Crop only the part that will remain in the result (more efficient)
    # Centered bbox of the final desired size in resized (larger/smaller) image coordinates
    y1, x1 = max(0, new_height - height) // 2, max(0, new_width - width) // 2
    y2, x2 = y1 + height, x1 + width
    bbox = np.array([y1,x1,y2,x2])
    # Map back to original image coordinates
    bbox = (bbox / zoom_factor).astype(np.int)
    y1, x1, y2, x2 = bbox
    cropped_img = img[y1:y2, x1:x2]

    # Handle padding when downscaling
    resize_height, resize_width = min(new_height, height), min(new_width, width)
    pad_height1, pad_width1 = (height - resize_height) // 2, (width - resize_width) //2
    pad_height2, pad_width2 = (height - resize_height) - pad_height1, (width - resize_width) - pad_width1
    pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2)] + [(0,0)] * (img.ndim - 2)

    result = cv2.resize(cropped_img, (resize_width, resize_height))
    result = np.pad(result, pad_spec, mode='constant')
    assert result.shape[0] == height and result.shape[1] == width
    return result

def translate(img, hpixels, vpixels):
    #img = cv2.imread('images/input.jpg')
    num_rows, num_cols = img.shape[:2]

    translation_matrix = np.float32([ [1,0,hpixels], [0,1,vpixels] ])
    new_img = cv2.warpAffine(img, translation_matrix, (num_cols, num_rows))
    return new_img



def data_augmentation(IMG_DIR, CLASSES, NUM_CLASSES):
    target_count = 1500
    split = 'train'


    for j in range(1, NUM_CLASSES):
        print(j, CLASSES[j])
        DIR = IMG_DIR + split + "/" + CLASSES[j]
        file_lst = [name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]
        random.shuffle(file_lst)

        itr = 0
        ctr = 0
        while len(file_lst) + ctr < target_count:
            file_dir = DIR + "/" + file_lst[itr]
            filename = file_dir.rsplit('/')[-1]
            filename = filename.replace('.jpg', '')
            #print(file_dir, filename)
            img = cv2.imread(file_dir)

            # rotation
            if random.random() > 0.6:
                angle = random.randint(-10, 10)
                rot_img = rotate(img, angle)
                rotate_token = True
            else:
                rot_img = img
                rotate_token = False

            # translation
            if random.random() > 0.6:
                htranslate = random.randint(-10, 10)
                vtranslate = random.randint(-10, 10)
                trans_img = translate(rot_img, htranslate, vtranslate)
            else:
                trans_img = rot_img

            NEW_IMG_DIR = DIR + "/" + filename + "_" + str(ctr) + ".jpg"

            if rotate_token:
                plt.imsave(NEW_IMG_DIR, trans_img[:,:,::-1])
            else:
                cv2.imwrite(NEW_IMG_DIR, trans_img)

            ctr += 1
            itr += 1

            if itr == len(file_lst):
                itr = 0
    return


'''
def data_augmentation(IMG_DIR, CLASSES, NUM_CLASSES):
    # augment data for training dataset only
    split = "train"
    for j in range(1, NUM_CLASSES): # I did not augment the 'none' images already
        DIR = IMG_DIR + split + "/" + CLASSES[j]
        file_lst = [name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]
        for file in file_lst:
            file_dir = DIR + "/" + file
            img = cv2.imread(file_dir)
            flip_img = flip(img)
            rot_img1 = rotate(img, angle=-20)
            rot_img2 = rotate(img, angle=-10)
            rot_img3 = rotate(img, angle=10)
            rot_img4 = rotate(img, angle=20)
            zoom_out_img = scale(img, zoom_factor = 0.8)
            zoom_in_img = scale(img, zoom_factor = 1.2)
            trans_left_img = translate(img,-10,0)
            trans_right_img = translate(img,10,0)
            trans_up_img = translate(img, 0, -10)
            trans_down_img = translate(img, 0, 10)
            
            #save new data
            filename = file[:-4]
            NEW_IMG_DIR = DIR + "/" + filename + "_"
            plt.imsave(NEW_IMG_DIR + 'rotate-10.jpg', rot_img2[:,:,::-1])
            plt.imsave(NEW_IMG_DIR +'rotate10.jpg', rot_img3[:,:,::-1])
            cv2.imwrite(NEW_IMG_DIR +'zoomout.jpg', zoom_out_img)
            cv2.imwrite(NEW_IMG_DIR +'zoomin.jpg', zoom_in_img)
            cv2.imwrite(NEW_IMG_DIR +'translate_left.jpg', trans_left_img)
            cv2.imwrite(NEW_IMG_DIR +'translate_right.jpg', trans_right_img)
            cv2.imwrite(NEW_IMG_DIR +'translate_up.jpg', trans_up_img)
            cv2.imwrite(NEW_IMG_DIR +'translate_down.jpg', trans_down_img)
            cv2.imwrite(NEW_IMG_DIR +'flip.jpg', flip_img)
            plt.imsave(NEW_IMG_DIR +'rotate-20.jpg', rot_img1[:,:,::-1])
            plt.imsave(NEW_IMG_DIR +'rotate20.jpg', rot_img4[:,:,::-1])
        
        print("Done augmenting data for label " + str(j))
'''

def process_data(IMG_DIR, CLASSES, NUM_CLASSES, split):
    
    proc_images = []
    labels = []
    for j in range(NUM_CLASSES):
        DIR = IMG_DIR + split + "/" + CLASSES[j]
        file_lst = [name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]
        for img_id in file_lst:
            img = cv2.imread(DIR + "/" + img_id)
            proc_images.append(img)
            labels.append(j)

        print('Done for class ' + str(j))
  
    proc_images = np.array(proc_images)
    labels = np.array(labels)
  
    # shuffle images and labels in unison
    proc_images, labels = shuffle(proc_images, labels, random_state=0)

    # one-hot encode the labels
    one_hot_labels = np.zeros((labels.size, labels.max()+1))
    one_hot_labels[np.arange(labels.size),labels] = 1

    np.save("{}_x.npy".format(split), proc_images)
    np.save("{}_y.npy".format(split), one_hot_labels)

    return proc_images, one_hot_labels

def color_preprocessing(x_split):
    x_split = x_split.astype('float32')

    x_split[:, :, :, 0] = (x_split[:, :, :, 0] - np.mean(x_split[:, :, :, 0])) / np.std(x_split[:, :, :, 0])
    x_split[:, :, :, 1] = (x_split[:, :, :, 1] - np.mean(x_split[:, :, :, 1])) / np.std(x_split[:, :, :, 1])
    x_split[:, :, :, 2] = (x_split[:, :, :, 2] - np.mean(x_split[:, :, :, 2])) / np.std(x_split[:, :, :, 2])

    return x_split

def combine_and_shuffle(DATA_DIR, CLASSES, NUM_CLASSES):
  
    DIR = DATA_DIR + CLASSES[0] + "_x.npy"
    proc_images = np.load(DIR)
    labels = np.repeat(0, len(proc_images))

    print('Done for class 0')

    for j in range(1, NUM_CLASSES):
        DIR = DATA_DIR + CLASSES[j] + "_x.npy"
        class_mat = np.load(DIR)
        proc_images = np.concatenate((proc_images, class_mat))
        label_arr =np.repeat(j, len(class_mat))
        labels = np.concatenate((labels, label_arr))

        print('Done for class ' + str(j))

    # shuffle images and labels in unison
    proc_images, labels = shuffle(proc_images, labels, random_state=0)

    # one-hot encode the labels
    one_hot_labels = np.zeros((labels.size, labels.max()+1))
    one_hot_labels[np.arange(labels.size),labels] = 1

    np.save("./data/train_x.npy", proc_images)
    np.save("./data/train_y.npy", one_hot_labels)

    return proc_images, one_hot_labels

def process_images(DATA_DIR, CLASSES, NUM_CLASSES, start = 0):

    for j in range(start, NUM_CLASSES):
        DIR = IMG_DIR + "train/" + CLASSES[j]
        file_lst = [name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]
        for img_id in file_lst:
            img = cv2.imread(DIR + "/" + img_id)
            filename = "{:02d}_{}".format(j, img_id) 
            np.save("./data/train/{}.npy".format(filename), img)

        print('Done for class ' + str(j))


def get_labels(DATA_DIR, CLASSES, NUM_CLASSES):

    image_labels = []
    for j in range(NUM_CLASSES):
        DIR = IMG_DIR + "train/" + CLASSES[j]
        file_lst = [name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]
        for img_id in file_lst:
            filename = "{:02d}_{}".format(j, img_id) 
            image_labels.append(filename)

        print('Done for class ' + str(j))

    with open('./data/train_labels.txt', 'w') as f:
        for item in image_labels:
            f.write("%s\n" % item)
    
    return image_labels



# Get data summary
#data_summary = get_image_count(IMG_DIR, CLASSES, NUM_CLASSES)
#print('Total number of images: ' + str(data_summary['total'].sum()))

#augment data
#data_augmentation(IMG_DIR, CLASSES, NUM_CLASSES)


#train_x, train_y = process_data(IMG_DIR, CLASSES, NUM_CLASSES, 'train')
#test_x, test_y = process_data(IMG_DIR, CLASSES, NUM_CLASSES, 'test')

#process_images(DATA_DIR, CLASSES, NUM_CLASSES, start = 0)
#get_labels(DATA_DIR, CLASSES, NUM_CLASSES)



# data_augmentation(IMG_DIR, CLASSES, NUM_CLASSES)
#process_images(DATA_DIR, CLASSES, NUM_CLASSES, start = 0)
#get_labels(DATA_DIR, CLASSES, NUM_CLASSES)
