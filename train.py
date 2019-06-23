import os
import argparse
import numpy as np
from keras.optimizers import SGD
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import plot_model
from keras.callbacks import LearningRateScheduler
from model import BiSeNet_ResNet18, BiSeNet_ResNet50, BiSeNet_Xception
from utils.generator import data_gen_small,read_labeled_image_list

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
data_dir = 'G:/Datasets/Cityscapes/leftImg8bit_trainvaltest'
data_list = 'G:/Datasets/Cityscapes/cityscapes_train_list.txt'
validate_list = 'G:/Datasets/Cityscapes/cityscapes_val_list.txt'

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=2, help='input batch size')
parser.add_argument('--learning_rate', type=float, default=1e-2, help='learning rate of the optimizer')
parser.add_argument('--decay', type=float, default=0.9, help='learning rate decay (per epoch)')
parser.add_argument('--epoch_steps', type=int, default=1487, help='steps/epoch to start training')#2975 for training
parser.add_argument('--n_epochs', type=int, default=60, help='number of epochs to train for')
parser.add_argument('--checkpoint', type=str, default=None, help='path to model checkpoint to resume training')
parser.add_argument("--input_shape", default=(512, 1024, 3), help="Input images shape")
parser.add_argument("--n_classes", default=19, type=int, help="Number of label")
# parser.add_argument("--val_steps", default=362, type=int, help="number of valdation step")
args = parser.parse_args()
print(args)


def PolyDecay(epochs):

    initial_lr = args.learning_rate
    power = 0.9
    n_epochs = args.n_epochs

    return initial_lr * np.power(1.0 - 1.0 * epochs / n_epochs, power)


# Callbacks
checkpoint = ModelCheckpoint('snapshots/BiSeNet_Xception_Resize_Weights_{epoch:03d}_{loss:.4f}.h5')
lr_decay = LearningRateScheduler(PolyDecay,verbose=1)
# tensorboard = TensorBoard(batch_size=args.batch_size)

# Generators
train_images, train_labels = read_labeled_image_list(data_dir, data_list)
train_gen = data_gen_small(train_images, train_labels, args.batch_size,
                           [args.input_shape[0], args.input_shape[1]], args.n_classes)

# Model
model_name = 'BiSeNet_Xception'
model = BiSeNet_Xception(args.input_shape, args.n_classes, training=True)

# Optimizer
optim = SGD(lr=args.learning_rate, momentum=0.9, decay=0.0005)
model.compile(optim, 'categorical_crossentropy', loss_weights=[1.0, 1.0, 1.0], metrics=['categorical_accuracy'])
model.summary()
# plot_model(model, show_shapes=True, to_file=model_name+'.png')

# # #Weights
weight_file = './snapshots/BiSeNet_Xception_Resize_Weights_030_0.2681.h5'
print('model == ', model_name)
print('Loading weights from ', weight_file)
model.load_weights(weight_file, by_name=True)

#Train
model.fit_generator(train_gen, steps_per_epoch=args.epoch_steps, epochs=args.n_epochs,
                    initial_epoch=0, callbacks=[checkpoint,lr_decay])






































