import os
import numpy as np
from inference import inference
from keras.preprocessing.image import *
from utils.generator import read_labeled_image_list


def calculate_iou(n_classes, pred_dir, val_labels):
    #建立一个方阵，对角线上求和为交集，一横一竖求和为并集
    conf_m = np.zeros((n_classes, n_classes), dtype=float)
    total = 0
    # mean_acc = 0.
    for i in range(len(val_labels)):
        print(i)
        total += 1
        res_dir = os.path.join(pred_dir, str(i) + '.png')
        pred = img_to_array(load_img(path=res_dir, target_size=(img_shape[0], img_shape[1])))
        label = img_to_array(load_img(path=val_labels[i], target_size=(img_shape[0], img_shape[1])))

        flat_pred = np.ravel(pred).astype(int)
        flat_label = np.ravel(label).astype(int)
        # acc = 0.
        # print(n_classes)
        for p, l in zip(flat_pred, flat_label):
            # print(p.type)
            # print(l)
            if l == 255:
                continue
            if l < n_classes and p < n_classes:
                conf_m[l, p] += 1
            else:
                print('Invalid entry encountered, skipping! Label: ', l,
                      ' Prediction: ', p, ' Img_num: ', str(i))
        #    if l==p:
        #        acc+=1
        #acc /= flat_pred.shape[0]
        #mean_acc += acc
    #mean_acc /= total
    #print 'mean acc: %f'%mean_acc
    I = np.diag(conf_m)
    U = np.sum(conf_m, axis=0) + np.sum(conf_m, axis=1) - I
    IOU = I/U
    meanIOU = np.mean(IOU)
    return conf_m, IOU, meanIOU




if __name__ == '__main__':

    n_classes = 19
    batch_size = 1
    img_shape = [512, 1024, 3]
    pred_dir = './pred/'
    dataset = 'Cityscapes'

    if dataset == 'Cityscapes':
        data_dir = 'G:/Datasets/Cityscapes/leftImg8bit_trainvaltest'
        train_list = 'G:/Datasets/Cityscapes/cityscapes_train_list.txt'
        val_list = 'G:/Datasets/Cityscapes/cityscapes_val_list.txt'

    model_name = 'BiSeNet_Xception'
    val_images, val_labels = read_labeled_image_list(data_dir, val_list)

    inference(model_name, n_classes, img_shape, val_images, val_labels, pred_dir, save_with_color=False)

    conf_m, IOU, meanIOU = calculate_iou(n_classes, pred_dir, val_labels)
    print('IOU: ')
    for i in range(len(IOU)):
        print(IOU[i])
    #meanIOU:
    print('%f' % meanIOU)
    print(np.sum(np.diag(conf_m)))
    print(np.sum(conf_m))
    #pixel acc:
    print('%f' % (np.sum(np.diag(conf_m))/np.sum(conf_m)))
