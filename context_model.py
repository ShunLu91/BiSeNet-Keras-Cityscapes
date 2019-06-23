from keras.layers import *
from keras import layers


def VGG16():
    def f(input):
        #1/2-conv3-64 *2
        y2 = Conv2D(64, 3, strides=1, padding='same', activation='relu', name='block1_conv1')(input)
        y2 = Conv2D(64, 3, strides=1, padding='same', activation='relu', name='block1_conv2')(y2)
        y2 = MaxPooling2D(pool_size=3, strides=2, padding='same', name='block1_pool')(y2)
        print('y0:',y2.shape)

        # 1/4-conv3-128 *2
        y2 = Conv2D(256, 3, strides=1, padding='same', activation='relu', name='block2_conv1')(y2)
        y2 = Conv2D(256, 3, strides=1, padding='same', activation='relu', name='block2_conv2')(y2)
        y2 = MaxPooling2D(pool_size=3, strides=2, padding='same', name='block2_pool')(y2)
        print('y1:',y2.shape)

        # 1/8-conv3-256 *3
        y2 = Conv2D(512, 3, strides=1, padding='same', activation='relu', name='block3_conv1')(y2)
        y2 = Conv2D(512, 3, strides=1, padding='same', activation='relu', name='block3_conv2')(y2)
        y2 = Conv2D(512, 3, strides=1, padding='same', activation='relu', name='block3_conv3')(y2)
        y2 = MaxPooling2D(pool_size=3, strides=2, padding='same', name='block3_pool')(y2)
        print('y2:',y2.shape)

        # 1/16-conv3-512 *3
        y2_16 = Conv2D(1024, 3, strides=1, padding='same', activation='relu', name='block4_conv1')(y2)
        y2_16 = Conv2D(1024, 3, strides=1, padding='same', activation='relu', name='block4_conv2')(y2_16)
        y2_16 = Conv2D(1024, 3, strides=1, padding='same', activation='relu', name='block4_conv3')(y2_16)
        y2_16 = MaxPooling2D(pool_size=3, strides=2, padding='same', name='block4_pool')(y2_16)
        print('y2_16:',y2_16.shape)

        # 1/32-conv3-1024 *3
        y2_32 = Conv2D(2048, 3, strides=1, padding='same', activation='relu', name='block5_conv1')(y2_16)
        y2_32 = Conv2D(2048, 3, strides=1, padding='same', activation='relu', name='block5_conv2')(y2_32)
        y2_32 = Conv2D(2048, 3, strides=1, padding='same', activation='relu', name='block5_conv3')(y2_32)
        y2_32 = MaxPooling2D(pool_size=3, strides=2, padding='same', name='block5_pool')(y2_32)
        print('y2_32:',y2_32.shape)

        # Global Pooling
        y2_Global = GlobalMaxPooling2D(data_format='channels_last')(y2_32)
        return y2_16, y2_32, y2_Global
    return f


def Xception():
    def f(input):
        # Block 1
        x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False, name='convolution2d_1')(input)
        x = BatchNormalization(name='batchnormalization_1')(x)
        x = Activation('relu', name='activation_1')(x)
        x = Conv2D(64, (3, 3), use_bias=False, name='convolution2d_2')(x)
        x = BatchNormalization(name='batchnormalization_2')(x)
        x = Activation('relu', name='activation_2')(x)

        residual = Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)

        # Block 2
        x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='separableconvolution2d_1')(x)
        x = BatchNormalization(name='batchnormalization_4')(x)
        x = Activation('relu', name='activation_3')(x)
        x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='separableconvolution2d_2')(x)
        x = BatchNormalization(name='batchnormalization_5')(x)

        # Block 2 Pool
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='maxpooling2d_1')(x)
        x = layers.add([x, residual],name='merge_1')

        residual = Conv2D(256, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)

        # Block 3
        x = Activation('relu',name='activation_4')(x)
        x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='separableconvolution2d_3')(x)
        x = BatchNormalization(name='batchnormalization_7')(x)
        x = Activation('relu', name='activation_5')(x)
        x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='separableconvolution2d_4')(x)
        x = BatchNormalization(name='batchnormalization_8')(x)

        # Block 3 Pool
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='maxpooling2d_2')(x)
        x = layers.add([x, residual], name='merge_2' )

        residual = Conv2D(728, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)

        # Block 4
        x = Activation('relu', name='activation_6')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='separableconvolution2d_5')(x)
        x = BatchNormalization(name='batchnormalization_10' )(x)
        x = Activation('relu', name='activation_7')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='separableconvolution2d_6' )(x)
        x = BatchNormalization(name='batchnormalization_11' )(x)

        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='maxpooling2d_3' )(x)
        x = layers.add([x, residual], name='merge_3' )

        # Block 5 - 12
        for i in range(8):
            #name
            activation_name0 = 'activation_%s' % (str(3*i+8))
            activation_name1 = 'activation_%s' % (str(3*i+9))
            activation_name2 = 'activation_%s' % (str(3*i+10))
            separable_name0 = 'separableconvolution2d_%s' % (str(3*i+7))
            separable_name1 = 'separableconvolution2d_%s' % (str(3*i+8))
            separable_name2 = 'separableconvolution2d_%s' % (str(3*i+9))
            batchnormalization_name0 = 'batchnormalization_%s' % (str(3*i+12))
            batchnormalization_name1 = 'batchnormalization_%s' % (str(3*i+13))
            batchnormalization_name2 = 'batchnormalization_%s' % (str(3*i+14))

            residual = x

            x = Activation('relu', name=activation_name0)(x)
            x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=separable_name0)(x)
            x = BatchNormalization(name=batchnormalization_name0)(x)
            x = Activation('relu', name=activation_name1)(x)
            x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=separable_name1)(x)
            x = BatchNormalization(name=batchnormalization_name1)(x)
            x = Activation('relu', name=activation_name2)(x)
            x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=separable_name2)(x)
            x = BatchNormalization(name=batchnormalization_name2)(x)

            x = layers.add([x, residual])

        residual = Conv2D(1024, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)

        # ========EXIT FLOW============
        # Block 13
        x = Activation('relu', name='activation_32')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='separableconvolution2d_31')(x)
        x = BatchNormalization(name='batchnormalization_37')(x)
        x = Activation('relu', name='activation_33')(x)
        x = SeparableConv2D(1024, (3, 3), padding='same', use_bias=False, name='separableconvolution2d_32')(x)
        x = BatchNormalization(name='batchnormalization_38')(x)
        y2_16 = x

        # Block 13 Pool
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='maxpooling2d_4')(x)
        x = layers.add([x, residual], name='merge_12')


        # Block 14
        x = SeparableConv2D(1536, (3, 3), padding='same', use_bias=False, name='separableconvolution2d_33')(x)
        x = BatchNormalization(name='batchnormalization_39')(x)
        x = Activation('relu', name='activation_34')(x)

        # Block 14 part 2
        x = SeparableConv2D(2048, (3, 3), padding='same', use_bias=False, name='separableconvolution2d_34')(x)
        x = BatchNormalization(name='batchnormalization_40')(x)
        x = Activation('relu', name='activation_35' )(x)
        y2_32 = x

        y2_Global = GlobalMaxPooling2D(data_format='channels_last', name='globalmaxpooling2d_1')(x)

        return y2_16, y2_32, y2_Global
    return f


def Xception1():
    def f(input):
        x = layers.Conv2D(32, (3, 3),
                          strides=(2, 2),
                          use_bias=False,
                          name='block1_conv1')(input)
        x = layers.BatchNormalization(name='block1_conv1_bn')(x)
        x = layers.Activation('relu', name='block1_conv1_act')(x)
        x = layers.Conv2D(64, (3, 3), use_bias=False, name='block1_conv2')(x)
        x = layers.BatchNormalization(name='block1_conv2_bn')(x)
        x = layers.Activation('relu', name='block1_conv2_act')(x)

        residual = layers.Conv2D(128, (1, 1),
                                 strides=(2, 2),
                                 padding='same',
                                 use_bias=False)(x)
        residual = layers.BatchNormalization()(residual)

        x = layers.SeparableConv2D(128, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name='block2_sepconv1')(x)
        x = layers.BatchNormalization(name='block2_sepconv1_bn')(x)
        x = layers.Activation('relu', name='block2_sepconv2_act')(x)
        x = layers.SeparableConv2D(128, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name='block2_sepconv2')(x)
        x = layers.BatchNormalization(name='block2_sepconv2_bn')(x)

        x = layers.MaxPooling2D((3, 3),
                                strides=(2, 2),
                                padding='same',
                                name='block2_pool')(x)
        x = layers.add([x, residual])

        residual = layers.Conv2D(256, (1, 1), strides=(2, 2),
                                 padding='same', use_bias=False)(x)
        residual = layers.BatchNormalization()(residual)

        x = layers.Activation('relu', name='block3_sepconv1_act')(x)
        x = layers.SeparableConv2D(256, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name='block3_sepconv1')(x)
        x = layers.BatchNormalization(name='block3_sepconv1_bn')(x)
        x = layers.Activation('relu', name='block3_sepconv2_act')(x)
        x = layers.SeparableConv2D(256, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name='block3_sepconv2')(x)
        x = layers.BatchNormalization(name='block3_sepconv2_bn')(x)

        x = layers.MaxPooling2D((3, 3), strides=(2, 2),
                                padding='same',
                                name='block3_pool')(x)
        x = layers.add([x, residual])

        residual = layers.Conv2D(728, (1, 1),
                                 strides=(2, 2),
                                 padding='same',
                                 use_bias=False)(x)
        residual = layers.BatchNormalization()(residual)

        x = layers.Activation('relu', name='block4_sepconv1_act')(x)
        x = layers.SeparableConv2D(728, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name='block4_sepconv1')(x)
        x = layers.BatchNormalization(name='block4_sepconv1_bn')(x)
        x = layers.Activation('relu', name='block4_sepconv2_act')(x)
        x = layers.SeparableConv2D(728, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name='block4_sepconv2')(x)
        x = layers.BatchNormalization(name='block4_sepconv2_bn')(x)

        x = layers.MaxPooling2D((3, 3), strides=(2, 2),
                                padding='same',
                                name='block4_pool')(x)
        x = layers.add([x, residual])

        for i in range(8):
            residual = x
            prefix = 'block' + str(i + 5)

            x = layers.Activation('relu', name=prefix + '_sepconv1_act')(x)
            x = layers.SeparableConv2D(728, (3, 3),
                                       padding='same',
                                       use_bias=False,
                                       name=prefix + '_sepconv1')(x)
            x = layers.BatchNormalization(name=prefix + '_sepconv1_bn')(x)
            x = layers.Activation('relu', name=prefix + '_sepconv2_act')(x)
            x = layers.SeparableConv2D(728, (3, 3),
                                       padding='same',
                                       use_bias=False,
                                       name=prefix + '_sepconv2')(x)
            x = layers.BatchNormalization(name=prefix + '_sepconv2_bn')(x)
            x = layers.Activation('relu', name=prefix + '_sepconv3_act')(x)
            x = layers.SeparableConv2D(728, (3, 3),
                                       padding='same',
                                       use_bias=False,
                                       name=prefix + '_sepconv3')(x)
            x = layers.BatchNormalization(name=prefix + '_sepconv3_bn')(x)

            x = layers.add([x, residual])

        #Exit flow
        residual = layers.Conv2D(1024, (1, 1), strides=(2, 2),
                                 padding='same', use_bias=False)(x)
        residual = layers.BatchNormalization()(residual)

        x = layers.Activation('relu', name='block13_sepconv1_act')(x)
        x = layers.SeparableConv2D(728, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name='block13_sepconv1')(x)
        x = layers.BatchNormalization(name='block13_sepconv1_bn')(x)
        x = layers.Activation('relu', name='block13_sepconv2_act')(x)
        x = layers.SeparableConv2D(1024, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name='block13_sepconv2')(x)
        x = layers.BatchNormalization(name='block13_sepconv2_bn')(x)
        y2_16 = x

        x = layers.MaxPooling2D((3, 3),
                                strides=(2, 2),
                                padding='same',
                                name='block13_pool')(x)
        x = layers.add([x, residual])


        x = layers.SeparableConv2D(1536, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name='block14_sepconv1')(x)
        x = layers.BatchNormalization(name='block14_sepconv1_bn')(x)
        x = layers.Activation('relu', name='block14_sepconv1_act')(x)

        x = layers.SeparableConv2D(2048, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name='block14_sepconv2')(x)
        x = layers.BatchNormalization(name='block14_sepconv2_bn')(x)
        x = layers.Activation('relu', name='block14_sepconv2_act')(x)
        y2_32 = x

        y2_Global = GlobalMaxPooling2D(data_format='channels_last', name='globalmaxpooling2d_1')(x)

        return y2_16, y2_32, y2_Global
    return f


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters

    bn_axis = 3

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2)):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.

    # Returns
        Output tensor for the block.

    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    bn_axis = 3

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1), strides=strides,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size, padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = layers.Conv2D(filters3, (1, 1), strides=strides,
                             kernel_initializer='he_normal',
                             name=conv_name_base + '1')(input_tensor)
    shortcut = layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def ResNet18():
    def f(input):
        x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(input)
        x = layers.Conv2D(64, (7, 7),
                          strides=(2, 2),
                          padding='valid',
                          kernel_initializer='he_normal',
                          name='conv1')(x)
        x = layers.BatchNormalization(name='bn_conv1')(x)
        x = layers.Activation('relu')(x)
        x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
        x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
        x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')

        x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')

        x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
        y2_16 = x

        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')

        y2_32 = x
        y2_Global = GlobalMaxPooling2D(data_format='channels_last', name='globalmaxpooling2d_1')(x)

        return y2_16, y2_32, y2_Global
    return f


def ResNet50_1():
    def f(input):
        x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(input)
        x = layers.Conv2D(64, (7, 7),
                          strides=(2, 2),
                          padding='valid',
                          kernel_initializer='he_normal',
                          name='conv1')(x)
        x = layers.BatchNormalization(name='bn_conv1')(x)
        x = layers.Activation('relu')(x)
        x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
        x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
        x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
        x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

        x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

        x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
        y2_16 = x

        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

        y2_32 = x
        y2_Global = GlobalMaxPooling2D(data_format='channels_last', name='globalmaxpooling2d_1')(x)

        return y2_16, y2_32, y2_Global
    return f


