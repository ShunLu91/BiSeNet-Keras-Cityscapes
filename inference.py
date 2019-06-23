import os
import numpy as np
import time

from PIL import Image
import matplotlib.pyplot as plt
from demo import decode_labels
from model import BiSeNet_ResNet18, BiSeNet_ResNet50, BiSeNet_Xception
from keras.preprocessing.image import load_img,img_to_array

from keras.models import load_model
from utils.generator import read_labeled_image_list

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# np.set_printoptions(threshold=np.inf)

def inference(model_name, n_classes, img_shape, val_images, val_labels, pred_dir, save_with_color = False):

    # Model
    # model = load_model(filepath=weight_file)
    if model_name == 'BiSeNet_ResNet18':
        model = BiSeNet_ResNet18(img_shape, n_classes, training=False)
        # weight_file = './snapshots/BiSeNet_ResNet18_Resize_Weights_030_6.0658.h5'
        weight_file = './snapshots/19_Classes/ResNet18/BiSeNet_ResNet18_Resize_Weights_030_6.0658.h5'

    if model_name == 'BiSeNet_ResNet50':
        model = BiSeNet_ResNet50(img_shape, n_classes, training=False)
        weight_file = './snapshots/19_Classes/ResNet50/BiSeNet_ResNet50_Resize_Weights_030_0.2185.h5'

    if model_name == 'BiSeNet_Xception':
        model = BiSeNet_Xception(img_shape, n_classes, training=False)
        weight_file = './snapshots/BiSeNet_Xception_Resize_Weights_030_0.2681.h5'
        # weight_file = './snapshots/19_Classes/Xception/BiSeNet_Xception_Resize_Weights_030_0.2681.h5'


    # model.summary()
    # Weights
    print('model == ', model_name)
    print('Loading weights from ',weight_file)
    model.load_weights(weight_file, by_name=True)

    duration = []
    for i in range(len(val_images)):
        img = img_to_array(load_img(path=val_images[i], target_size=(img_shape[0], img_shape[1]),
                                    interpolation='bilinear')) / 255
        label = load_img(path=val_labels[i], target_size=(img_shape[0], img_shape[1]), color_mode="grayscale")
        label = np.squeeze(img_to_array(label,dtype='int64'))

        # predict
        x = np.expand_dims(img, axis=0)

        start = time.time()
        pred = model.predict(x=x)  # 输出是4维tensor
        end = time.time()
        if i >0:
            # print(end - start)
            duration.append(end-start)

        pred = np.argmax(np.squeeze(pred), axis=-1).astype('uint8')  # 输出是2维numpy数组

        # save prediction
        if save_with_color :
            inference_dir = './inference'
            label = decode_labels(label)
            pred = decode_labels(pred)

            fig, axes = plt.subplots(3, figsize=(16, 12))
            axes.flat[0 * 3].set_title('data')
            axes.flat[0 * 3].imshow(img)

            axes.flat[0 * 3 + 1].set_title('label')
            axes.flat[0 * 3 + 1].imshow(label)

            axes.flat[0 * 3 + 2].set_title('pred')
            axes.flat[0 * 3 + 2].imshow(pred)

            plt.savefig(os.path.join(inference_dir, 'Xception_Modified' + str(i) + '.png'))
            plt.close(fig)
        else :
            result_img = Image.fromarray(pred.astype('uint8'), mode='L')
            result_img.save(os.path.join(pred_dir, str(i) + '.png'))
        print('{} img successfully predicted'.format(i))
    fps = 1/(np.mean(duration))
    ms = 1000*np.mean(duration)
    print('The average time is {}ms'.format(ms))
    print('About {} fps'.format(fps))

if __name__ =='__main__':
    img_shape = [512, 1024, 3]
    pred_dir = './pred'
    n_classes = 19
    batch_size = 1
    dataset = 'Cityscapes'
    model_name = 'BiSeNet_Xception'

    if dataset == 'Cityscapes':
        data_dir = 'G:/Datasets/Cityscapes/leftImg8bit_trainvaltest'
        train_list = 'G:/Datasets/Cityscapes/cityscapes_train_list.txt'
        val_list = 'G:/Datasets/Cityscapes/cityscapes_val_list.txt'

    val_images, val_labels = read_labeled_image_list(data_dir, val_list)

    inference(model_name, n_classes, img_shape, val_images, val_labels, pred_dir, save_with_color=True)
