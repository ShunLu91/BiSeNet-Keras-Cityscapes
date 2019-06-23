import os
import cv2
import random
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import load_img,img_to_array

data_dir = 'G:/Datasets/Cityscapes/leftImg8bit_trainvaltest'
data_list = 'G:/Datasets/Cityscapes/cityscapes_train_list.txt'
validate_list = 'G:/Datasets/Cityscapes/cityscapes_val_list.txt'

def read_labeled_image_list(data_dir, data_list):
    f = open(data_list, 'r')
    images = []
    masks = []
    for line in f:
        try:
            image, mask = line[:-1].split(' ')
        except ValueError: # Adhoc for test.
            image = mask = line.strip("\n")
        image = os.path.join(data_dir, image)
        mask = os.path.join(data_dir, mask)
        if not tf.gfile.Exists(image):
            raise ValueError('Failed to find file: ' + image)
        if not tf.gfile.Exists(mask):
            raise ValueError('Failed to find file: ' + mask)
        images.append(image)
        masks.append(mask)
    return images, masks


def category_label(labels, dims, n_labels, ignore_label=None):
    x = np.zeros([dims[0], dims[1], n_labels])
    for i in range(dims[0]):
        for j in range(dims[1]):
            # print('i:',i)
            # print('j:',j)
            if labels[i][j] == 255:
                continue
            else:
                x[i, j, labels[i][j]] = 1
    # x = x.reshape(dims[0], dims[1], n_labels)
    # print('x_shape：',x.shape)
    return x

# def Ignore_Label(labels, dims, ignore):
#     for i in range(dims[0]):
#         for j in range(dims[1]):
#             if labels[i][j] == ignore:
#                 labels[i][j] = 19
#     # x = x.reshape(dims[0], dims[1], n_labels)
#     # print('x_shape：',x.shape)
#     return labels

# generator that we will use to read the data from the directory
def data_gen_small(image_list, label_list, batch_size, dims, n_classes, random_crop=False, resize_nearst=True):

    while True:
        ix = np.random.choice(np.arange(len(image_list)), batch_size)
        images = []
        labels = []
        for i in ix:
            # random_crop
            if random_crop:
                image_path = image_list[i]
                image = img_to_array(load_img(image_path,interpolation='bilinear')) / 255
                label_path = label_list[i]
                label = img_to_array(load_img(label_path, color_mode="grayscale"), dtype='int32')
                y = random.randint(0, image.shape[0] - dims[0])
                x = random.randint(0, image.shape[1] - dims[1])
                crop_image = image[(y):(y + dims[0]), (x):(x + dims[1])]
                crop_label = label[(y):(y + dims[0]), (x):(x + dims[1])]
                crop_label = category_label(crop_label, (dims[0], dims[1]), n_classes)
                images.append(crop_image)
                labels.append(crop_label)
            #resize
            if resize_nearst:
                # images
                image_path = image_list[i]
                image = load_img(image_path,target_size=(dims[0], dims[1]))
                image = img_to_array(image) / 255
                images.append(image)
                # labels
                label_path = label_list[i]
                label = load_img(label_path,target_size=(dims[0], dims[1]), color_mode="grayscale")
                label = img_to_array(label,dtype='int32')
                label = category_label(label, (dims[0], dims[1]), n_classes) #label to one-hot
                labels.append(label)
        images = np.array(images)
        labels = np.array(labels)
        yield images, [labels, labels, labels]

if __name__ == '__main__':
    train_images, train_masks = read_labeled_image_list(data_dir, data_list)
    while True:
        # print(1)
        train_gen = data_gen_small(train_images, train_masks, 4,[256, 256], 19)