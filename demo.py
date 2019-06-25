import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from model import BiSeNet_ResNet50, BiSeNet_Xception
from keras.preprocessing import image
from keras.models import load_model
from utils.color import Cityscapes_Colors

np.set_printoptions(threshold=np.inf)

def decode_labels(mask):
    """Decode batch of segmentation masks.

    Args:
      label_batch: result of inference after taking argmax.

    Returns:
      An batch of RGB images of the same s ize
    """
    img = Image.new('RGB', size=(len(mask[0]), len(mask)))
    pixels = img.load()
    for j_, j in enumerate(mask):
        for k_, k in enumerate(j):
            if k < 19:
                pixels[k_, j_] = Cityscapes_Colors[k]
    return np.array(img)


if __name__ == '__main__':

    img_shape = [512, 512, 3]

    # restore model
    # weight_file_path = './snapshots/weights.003-0.2173.h5'
    # print('Loading model…')
    # model = load_model(filepath=weight_file_path)

    #restore weights
    model = BiSeNet_ResNet50(img_shape, 20, 1e-2)
    model.summary()
    weight_file_path = './snapshots/Xception/weights.015-0.2004.h5'
    print('Loading weights…')
    model.load_weights(weight_file_path, by_name=True)

    #read
    img_path = 'aachen_000000_000019_leftImg8bit.png'
    img = image.load_img(path=img_path, target_size=(img_shape[0], img_shape[1]))
    x = image.img_to_array(img)/255

    #predict
    x = np.expand_dims(x, axis=0)
    start_time = time.time()
    pred = model.predict(x=x)
    duration = time.time() - start_time
    print('{}s used to make predictions.\n'.format(duration))
    pred = np.squeeze(np.argmax(pred,axis=-1))#输出是2维numpy数组

    # convert prediction to color
    pred_image = decode_labels(pred)
    # print(pred[0,:,:])

    # show
    plt.imshow(pred_image)
    plt.title('pred_image')
    plt.axis('off')  # 不显示坐标轴
    plt.show()

