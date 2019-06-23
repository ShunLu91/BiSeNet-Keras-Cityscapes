# BiSeNet-Keras-Cityscapes
This repo is an implementation of BiSeNet in Keras on the [Cityscapes](https://www.cityscapes-dataset.com/) dataset. But it still has some differences compared with the author's.The code is currently available and will fix issues in the code later.

The context_model mainly contains ResNet18,ResNet50 and Xception which you can use the pretrained model provided by [Keras-Model](https://github.com/fchollet/deep-learning-models/releases).

##Training
You can initialize the model weights using the Keras pretrained model weights.

##Evaluation
Here I put my trained BiSeNet-Xception weights on the [Google Drive](https://drive.google.com/file/d/1AOc_50QrEC-ZYvTDHGDPWJEScvvxAxzH/view?usp=sharing) which can achieve 64.3% mIoU on the Cityscapes validation set.
