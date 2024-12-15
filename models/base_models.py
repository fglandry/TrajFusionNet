import os
import wget

from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.layers import Flatten, Dropout, Activation
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.layers import Conv3D, MaxPooling3D, ZeroPadding3D
from tensorflow.keras.layers import AveragePooling3D
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Lambda, BatchNormalization
from tensorflow.keras.layers import Conv1D, MultiHeadAttention, LayerNormalization, Dropout, GlobalAveragePooling1D # transformer imports
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Concatenate, concatenate, Dense
from tensorflow.keras.layers import Layer, Embedding, TimeDistributed
#import tensorflow.keras.backend as K
#import tensorflow as tf

from models.custom_layers import CustomAttention, T2V


def AlexNet(include_top=True,
            weights=None,
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=1000,
            classifier_activation='softmax'):
    '''
    Implementation of AlexNet based on the paper
    Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2017). Imagenet classification with 
    deep convolutional neural networks. Communications of the ACM, 60(6), 84-90.
    '''
    if input_shape is None:
        input_shape = (227, 227, 3)
    if input_tensor is None:
        inputs = Input(shape=input_shape)
    else:
        inputs = input_tensor

    x = inputs
    x = Conv2D(filters=96, kernel_size=11, strides=4, padding='valid', activation='relu')(x)
    x = MaxPooling2D(pool_size=3, strides=2, padding='valid')(x)
    x = ZeroPadding2D((2, 2))(x)
    x = Conv2D(filters=256, kernel_size=5, strides=1, padding='valid', activation='relu')(x)
    x = MaxPooling2D(pool_size=3, strides=2, padding='valid')(x)
    x = ZeroPadding2D((2, 2))(x)
    x = Conv2D(filters=384, kernel_size=3, strides=1, padding='valid', activation='relu')(x)
    x = BatchNormalization()(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(filters=384, kernel_size=3, strides=1, padding='valid', activation='relu')(x)
    x = BatchNormalization()(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(filters=256, kernel_size=3, strides=1, padding='valid', activation='relu')(x)
    x = MaxPooling2D(pool_size=3, strides=2, padding='valid')(x)
    x = BatchNormalization()(x)
    if include_top:
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(4096, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(classes, activation=classifier_activation)(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)
    model = Model(inputs, x, name='alexnet')

    # if weights == 'imagenet':
    #     model.load_weights('weights/alexnet_weights_tensorflow.h5', by_name=True)

    return model


def convert_to_fcn(model, classes=2, activation='softmax',
                   pooling='avg', features=False, model_type='alexnet'):
    """
    Converts a given CNN model to a FCN model
    Args:
        model: The model object
        classes: Number of classes
        activation: Type of activation for the last layer
        pooling: Pooling type for generating features
        features: Whether to return convolutional features or apply global pooling and activation
        model_type: The type of CNN. Support alexnet, vgg16, and resnet50
    Returns:
        Model object
    """
    num_filters = 4096
    if 'resnet' in model_type:
        num_filters = 2048
    x = Conv2D(filters=num_filters, kernel_size=(6, 6), strides=(1, 1), padding='valid')(model.output)
    x = Conv2D(filters=num_filters, kernel_size=(1, 1), strides=(1, 1), padding='valid')(x)
    x = Conv2D(filters=classes, kernel_size=(1, 1), strides=(1, 1), padding='valid')(x)

    if features:
        if pooling == 'avg':
            x = Lambda(lambda x: K.mean(x, axis=-1))(x)
        else:
            x = Lambda(lambda x: K.max(x, axis=-1))(x)
        x = Flatten(name='fcn_features')(x)
    else:
        x = GlobalMaxPooling2D()(x)
        x = Activation(activation)(x)
    return Model(model.input, x)


def C3DNet(freeze_conv_layers=False, weights=None,
           dense_activation='softmax', dropout=0.5, include_top=False,
           input_layer=None, *args, **kwargs):
    """
    C3D model implementation. Source: https://github.com/adamcasson/c3d
    Reference: Du Tran, Lubomir Bourdev, Rob Fergus, Lorenzo Torresani,and Manohar Paluri. 
    Learning spatiotemporal features with 3D convolutional networks. ICCV, 2015.
    Args:
        freeze_conv_layers: Whether to freeze convolutional layers at the time of training
        weights: Pre-trained weights
        dense_activation: Activation of the last layer
        dropout: Dropout of dense layers
        include_top: Whether to add fc layers
    Returns:
        C3D model
    """
    #if not input_layer:
    input_layer = Input(shape=(16, 112, 112, 3))
    all_inputs = input_layer
    """
    input_data = Input(shape=(16, 112, 112, 3))
    input_2 = Input(shape=(16, sequence_features_len))
    #input_2 = Input(shape=(16, 4))
    #input_3 = Input(shape=(16, 1))
    #input_4 = Input(shape=(16, 72))
    all_inputs = [input_data, input_2]
    #input_data = tf.gather(input_data, 0)
    """
    model = Conv3D(64, 3, activation='relu', padding='same', name='conv1')(input_layer)
    model = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='valid', name='pool1')(model)
    # 2nd layer group
    model = Conv3D(128, 3, activation='relu', padding='same', name='conv2')(model)
    model = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool2')(model)
    # 3rd layer group
    model = Conv3D(256, 3, activation='relu', padding='same', name='conv3a')(model)
    model = Conv3D(256, 3, activation='relu', padding='same', name='conv3b')(model)
    model = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool3')(model)
    # 4th layer group
    model = Conv3D(512, 3, activation='relu', padding='same', name='conv4a')(model)
    model = Conv3D(512, 3, activation='relu', padding='same', name='conv4b')(model)
    model = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool4')(model)
    # 5th layer group
    model = Conv3D(512, 3, activation='relu', padding='same', name='conv5a')(model)
    model = Conv3D(512, 3, activation='relu', padding='same', name='conv5b')(model)
    model = ZeroPadding3D(padding=(0, 1, 1), name='zeropad5')(model)  # ((0, 0), (0, 1), (0, 1))
    model = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool5')(model)
    model_flatten = Flatten(name='flatten')(model)

    # # FC layers group
    model = Dense(4096, activation='relu', name='fc6')(model_flatten)
    model = Dropout(dropout)(model)
    model = Dense(4096, activation='relu', name='fc7')(model)
    model_fc7 = Dropout(dropout)(model)
    model_fc8 = Dense(487, activation=dense_activation, name='fc8')(model_fc7)

    net_model = Model(all_inputs, model_fc8)
    if weights is not None:
        net_model.load_weights(weights)

    if include_top:
        model_fc8_new = Dense(1, activation=dense_activation, name='fc8')(model_fc7)
        net_model = Model(all_inputs, model_fc8_new)
        if freeze_conv_layers:
            for layer in model.layers[:-5]:
                layer.trainable = False
            for layer in model.layers:
                print(layer.name, layer.trainable)
    else:
        #net_model = Model(all_inputs, model_flatten)
        net_model = model_fc8

    return net_model, input_layer
    

def I3DNet(freeze_conv_layers=False, weights=None, classes=1,
           dense_activation='softmax', dropout=0.5, num_channels=3, include_top=False):
    """
    I3D model implementation. Source: https://github.com/dlpbc/keras-kinetics-i3d
    Reference: Joao Carreira and Andrew Zisserman.  Quo vadis, action recognition?
    A new model and the kinetics dataset. CVPR, 2017.
    Args:
        freeze_conv_layers: Whether to freeze convolutional layers at the time of training
        weights: Pre-trained weights
        classes: Number of classes
        dense_activation: Activation of the last layer
        dropout: Dropout of dense layers
        include_top: Whether to add fc layers
    Returns:
        I3D model
    """
    def conv3d_bn(x,
                  filters,
                  num_frames,
                  num_row,
                  num_col,
                  padding='same',
                  strides=(1, 1, 1),
                  use_bias=False,
                  use_activation_fn=True,
                  use_bn=True,
                  name=None):
        """Utility function to apply conv3d + BN.

        # Arguments
            x: input tensor.
            filters: filters in `Conv3D`.
            num_frames: frames (time depth) of the convolution kernel.
            num_row: height of the convolution kernel.
            num_col: width of the convolution kernel.
            padding: padding mode in `Conv3D`.
            strides: strides in `Conv3D`.
            use_bias: use bias or not
            use_activation_fn: use an activation function or not.
            use_bn: use batch normalization or not.
            name: name of the ops; will become `name + '_conv'`
                for the convolution and `name + '_bn'` for the
                batch norm layer.

        # Returns
            Output tensor after applying `Conv3D` and `BatchNormalization`.
        """
        if name is not None:
            bn_name = name + '_bn'
            conv_name = name + '_conv'
        else:
            bn_name = None
            conv_name = None

        x = Conv3D(
            filters, (num_frames, num_row, num_col),
            strides=strides,
            padding=padding,
            use_bias=use_bias,
            name=conv_name)(x)

        if use_bn:
            bn_axis = 4
            x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)

        if use_activation_fn:
            x = Activation('relu', name=name)(x)

        return x

    channel_axis = 4
    inputs = Input(shape=(16, 224, 224, num_channels))

    # Downsampling via convolution (spatial and temporal)
    x = conv3d_bn(inputs, 64, 7, 7, 7, strides=(2, 2, 2), padding='same', name='Conv3d_1a_7x7')

    # Downsampling (spatial only)
    x = MaxPooling3D((1, 3, 3), strides=(1, 2, 2), padding='same', name='MaxPool2d_2a_3x3')(x)
    x = conv3d_bn(x, 64, 1, 1, 1, strides=(1, 1, 1), padding='same', name='Conv3d_2b_1x1')
    x = conv3d_bn(x, 192, 3, 3, 3, strides=(1, 1, 1), padding='same', name='Conv3d_2c_3x3')

    # Downsampling (spatial only)
    x = MaxPooling3D((1, 3, 3), strides=(1, 2, 2), padding='same', name='MaxPool2d_3a_3x3')(x)

    # Mixed 3b
    branch_0 = conv3d_bn(x, 64, 1, 1, 1, padding='same', name='Conv3d_3b_0a_1x1')

    branch_1 = conv3d_bn(x, 96, 1, 1, 1, padding='same', name='Conv3d_3b_1a_1x1')
    branch_1 = conv3d_bn(branch_1, 128, 3, 3, 3, padding='same', name='Conv3d_3b_1b_3x3')

    branch_2 = conv3d_bn(x, 16, 1, 1, 1, padding='same', name='Conv3d_3b_2a_1x1')
    branch_2 = conv3d_bn(branch_2, 32, 3, 3, 3, padding='same', name='Conv3d_3b_2b_3x3')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_3b_3a_3x3')(x)
    branch_3 = conv3d_bn(branch_3, 32, 1, 1, 1, padding='same', name='Conv3d_3b_3b_1x1')

    x = Concatenate(axis=channel_axis, name='Mixed_3b')([branch_0, branch_1, branch_2, branch_3])

    # Mixed 3c
    branch_0 = conv3d_bn(x, 128, 1, 1, 1, padding='same', name='Conv3d_3c_0a_1x1')

    branch_1 = conv3d_bn(x, 128, 1, 1, 1, padding='same', name='Conv3d_3c_1a_1x1')
    branch_1 = conv3d_bn(branch_1, 192, 3, 3, 3, padding='same', name='Conv3d_3c_1b_3x3')

    branch_2 = conv3d_bn(x, 32, 1, 1, 1, padding='same', name='Conv3d_3c_2a_1x1')
    branch_2 = conv3d_bn(branch_2, 96, 3, 3, 3, padding='same', name='Conv3d_3c_2b_3x3')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_3c_3a_3x3')(x)
    branch_3 = conv3d_bn(branch_3, 64, 1, 1, 1, padding='same', name='Conv3d_3c_3b_1x1')

    x = Concatenate(axis=channel_axis, name='Mixed_3c')([branch_0, branch_1, branch_2, branch_3])

    # Downsampling (spatial and temporal)
    x = MaxPooling3D((3, 3, 3), strides=(2, 2, 2), padding='same', name='MaxPool2d_4a_3x3')(x)

    # Mixed 4b
    branch_0 = conv3d_bn(x, 192, 1, 1, 1, padding='same', name='Conv3d_4b_0a_1x1')

    branch_1 = conv3d_bn(x, 96, 1, 1, 1, padding='same', name='Conv3d_4b_1a_1x1')
    branch_1 = conv3d_bn(branch_1, 208, 3, 3, 3, padding='same', name='Conv3d_4b_1b_3x3')

    branch_2 = conv3d_bn(x, 16, 1, 1, 1, padding='same', name='Conv3d_4b_2a_1x1')
    branch_2 = conv3d_bn(branch_2, 48, 3, 3, 3, padding='same', name='Conv3d_4b_2b_3x3')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_4b_3a_3x3')(x)
    branch_3 = conv3d_bn(branch_3, 64, 1, 1, 1, padding='same', name='Conv3d_4b_3b_1x1')

    x = Concatenate(axis=channel_axis, name='Mixed_4b')([branch_0, branch_1, branch_2, branch_3])

    # Mixed 4c
    branch_0 = conv3d_bn(x, 160, 1, 1, 1, padding='same', name='Conv3d_4c_0a_1x1')

    branch_1 = conv3d_bn(x, 112, 1, 1, 1, padding='same', name='Conv3d_4c_1a_1x1')
    branch_1 = conv3d_bn(branch_1, 224, 3, 3, 3, padding='same', name='Conv3d_4c_1b_3x3')

    branch_2 = conv3d_bn(x, 24, 1, 1, 1, padding='same', name='Conv3d_4c_2a_1x1')
    branch_2 = conv3d_bn(branch_2, 64, 3, 3, 3, padding='same', name='Conv3d_4c_2b_3x3')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_4c_3a_3x3')(x)
    branch_3 = conv3d_bn(branch_3, 64, 1, 1, 1, padding='same', name='Conv3d_4c_3b_1x1')

    x = Concatenate(axis=channel_axis, name='Mixed_4c')([branch_0, branch_1, branch_2, branch_3])

    # Mixed 4d
    branch_0 = conv3d_bn(x, 128, 1, 1, 1, padding='same', name='Conv3d_4d_0a_1x1')

    branch_1 = conv3d_bn(x, 128, 1, 1, 1, padding='same', name='Conv3d_4d_1a_1x1')
    branch_1 = conv3d_bn(branch_1, 256, 3, 3, 3, padding='same', name='Conv3d_4d_1b_3x3')

    branch_2 = conv3d_bn(x, 24, 1, 1, 1, padding='same', name='Conv3d_4d_2a_1x1')
    branch_2 = conv3d_bn(branch_2, 64, 3, 3, 3, padding='same', name='Conv3d_4d_2b_3x3')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_4d_3a_3x3')(x)
    branch_3 = conv3d_bn(branch_3, 64, 1, 1, 1, padding='same', name='Conv3d_4d_3b_1x1')

    x = Concatenate(axis=channel_axis, name='Mixed_4d')([branch_0, branch_1, branch_2, branch_3])

    # Mixed 4e
    branch_0 = conv3d_bn(x, 112, 1, 1, 1, padding='same', name='Conv3d_4e_0a_1x1')

    branch_1 = conv3d_bn(x, 144, 1, 1, 1, padding='same', name='Conv3d_4e_1a_1x1')
    branch_1 = conv3d_bn(branch_1, 288, 3, 3, 3, padding='same', name='Conv3d_4e_1b_3x3')

    branch_2 = conv3d_bn(x, 32, 1, 1, 1, padding='same', name='Conv3d_4e_2a_1x1')
    branch_2 = conv3d_bn(branch_2, 64, 3, 3, 3, padding='same', name='Conv3d_4e_2b_3x3')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_4e_3a_3x3')(x)
    branch_3 = conv3d_bn(branch_3, 64, 1, 1, 1, padding='same', name='Conv3d_4e_3b_1x1')

    x = Concatenate(axis=channel_axis, name='Mixed_4e')([branch_0, branch_1, branch_2, branch_3])

    # Mixed 4f
    branch_0 = conv3d_bn(x, 256, 1, 1, 1, padding='same', name='Conv3d_4f_0a_1x1')

    branch_1 = conv3d_bn(x, 160, 1, 1, 1, padding='same', name='Conv3d_4f_1a_1x1')
    branch_1 = conv3d_bn(branch_1, 320, 3, 3, 3, padding='same', name='Conv3d_4f_1b_3x3')

    branch_2 = conv3d_bn(x, 32, 1, 1, 1, padding='same', name='Conv3d_4f_2a_1x1')
    branch_2 = conv3d_bn(branch_2, 128, 3, 3, 3, padding='same', name='Conv3d_4f_2b_3x3')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_4f_3a_3x3')(x)
    branch_3 = conv3d_bn(branch_3, 128, 1, 1, 1, padding='same', name='Conv3d_4f_3b_1x1')

    x = Concatenate(axis=channel_axis, name='Mixed_4f')([branch_0, branch_1, branch_2, branch_3])

    # Downsampling (spatial and temporal)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same', name='MaxPool2d_5a_2x2')(x)

    # Mixed 5b
    branch_0 = conv3d_bn(x, 256, 1, 1, 1, padding='same', name='Conv3d_5b_0a_1x1')

    branch_1 = conv3d_bn(x, 160, 1, 1, 1, padding='same', name='Conv3d_5b_1a_1x1')
    branch_1 = conv3d_bn(branch_1, 320, 3, 3, 3, padding='same', name='Conv3d_5b_1b_3x3')

    branch_2 = conv3d_bn(x, 32, 1, 1, 1, padding='same', name='Conv3d_5b_2a_1x1')
    branch_2 = conv3d_bn(branch_2, 128, 3, 3, 3, padding='same', name='Conv3d_5b_2b_3x3')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_5b_3a_3x3')(x)
    branch_3 = conv3d_bn(branch_3, 128, 1, 1, 1, padding='same', name='Conv3d_5b_3b_1x1')

    x = Concatenate(axis=channel_axis, name='Mixed_5b')([branch_0, branch_1, branch_2, branch_3])

    # Mixed 5c
    branch_0 = conv3d_bn(x, 384, 1, 1, 1, padding='same', name='Conv3d_5c_0a_1x1')

    branch_1 = conv3d_bn(x, 192, 1, 1, 1, padding='same', name='Conv3d_5c_1a_1x1')
    branch_1 = conv3d_bn(branch_1, 384, 3, 3, 3, padding='same', name='Conv3d_5c_1b_3x3')

    branch_2 = conv3d_bn(x, 48, 1, 1, 1, padding='same', name='Conv3d_5c_2a_1x1')
    branch_2 = conv3d_bn(branch_2, 128, 3, 3, 3, padding='same', name='Conv3d_5c_2b_3x3')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_5c_3a_3x3')(x)
    branch_3 = conv3d_bn(branch_3, 128, 1, 1, 1, padding='same', name='Conv3d_5c_3b_1x1')

    x_concatenate = Concatenate(axis=channel_axis, name='Mixed_5c')([branch_0, branch_1, branch_2, branch_3])


    # create model
    if include_top:
        # Classification block
        x = AveragePooling3D((2, 7, 7), strides=(1, 1, 1), padding='valid',
                             name='global_avg_pool')(x_concatenate)
        x = Dropout(dropout)(x)
        x = conv3d_bn(x, classes, 1, 1, 1, padding='same',
                      use_bias=True, use_activation_fn=False,
                      use_bn=False, name='Conv3d_6a_1x1_new')
        num_frames_remaining = int(x.shape[1])
        x = Reshape((num_frames_remaining, classes))(x)
        # logits (raw scores for each class)
        x = Lambda(lambda x: K.mean(x, axis=1, keepdims=False),
                   output_shape=lambda s: (s[0], s[2]))(x)
        # if not endpoint_logit:
        x = Activation(dense_activation, name='prediction')(x)
        net_model = Model(inputs, x, name='i3d_inception')
        if freeze_conv_layers:
            for layer in net_model.layers[:-5]:
                layer.trainable = False
            # for layer in net_model.layers:
            #     print(layer.name, layer.trainable)
    else:
        h = int(x.shape[2])
        w = int(x.shape[3])
        x = AveragePooling3D((2, h, w), strides=(1, 1, 1), padding='valid', name='global_avg_pool')(x_concatenate)
        net_model = Model(inputs, x, name='i3d_no_top')
        if freeze_conv_layers:
            for layer in net_model.layers[:-5]:
                layer.trainable = False
            # for layer in net_model.layers:
            #     print(layer.name, layer.trainable)

    if weights is not None:
        net_model.load_weights(weights, by_name=True)

    return net_model

"""
class Time2Vec(tf.keras.layers.Layer):
    def __init__(self, kernel_size=1):
        super(Time2Vec, self).__init__(trainable=True, name='Time2VecLayer')
        self.k = kernel_size
    
    def build(self, input_shape):
        # trend
        self.wb = self.add_weight(name='wb',shape=(input_shape[1],),initializer='uniform',trainable=True)
        self.bb = self.add_weight(name='bb',shape=(input_shape[1],),initializer='uniform',trainable=True)
        # periodic
        self.wa = self.add_weight(name='wa',shape=(1, input_shape[1], self.k),initializer='uniform',trainable=True)
        self.ba = self.add_weight(name='ba',shape=(1, input_shape[1], self.k),initializer='uniform',trainable=True)
        super(Time2Vec, self).build(input_shape)
    
    def call(self, inputs, **kwargs):
        bias = self.wb * inputs + self.bb
        dp = K.dot(inputs, self.wa) + self.ba
        wgts = K.sin(dp) # or K.cos(.)

        ret = K.concatenate([K.expand_dims(bias, -1), wgts], -1)
        ret = K.reshape(ret, (-1, inputs.shape[1]*(self.k+1)))
        return ret
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1]*(self.k + 1))
"""

class TransformerNet():

    @staticmethod
    def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
        # Normalization and Attention
        x = LayerNormalization(epsilon=1e-6)(inputs)
        x = MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(x, x)
        x = Dropout(dropout)(x)
        res = x + inputs

        # Feed Forward Part
        x = LayerNormalization(epsilon=1e-6)(res)
        x = Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
        x = Dropout(dropout)(x)
        x = Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        return x + res
    
    """
    @staticmethod
    class TokenAndPositionEmbedding(Layer):
        def __init__(self, maxlen, vocab_size, embed_dim):
            super().__init__()
            self.token_emb = Embedding(input_dim=vocab_size, output_dim=embed_dim)
            self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim)

        def call(self, x):
            maxlen = tf.shape(x)[-1]
            positions = tf.range(start=0, limit=maxlen, delta=1)
            positions = self.pos_emb(positions)
            x = self.token_emb(x)
            return x + positions

    @staticmethod
    def embedding_layer(input):
        input_shape = input.get_shape().as_list()
        maxlen = input_shape[1]
        vocab_size = input_shape[2]
        embed_dim = 10
        embed_layer = TransformerNet.TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
        return embed_layer(input)
    """

    """
        time2vec = Time2Vec(kernel_size=1)
        time_embedding = tf.keras.layers.TimeDistributed(time2vec)(x)
        x = K.concatenate([inputs, time_embedding], -1)
        x = Time2Vec(inputs)
    """
        
    @staticmethod
    def build_model(
        model_opts,
        inputs,
        head_size,
        num_heads,
        ff_dim,
        num_transformer_blocks,
        mlp_units,
        dropout=0,
        mlp_dropout=0,
        n_classes=1,
        combined_model=False,
        include_top=True
    ):
        if combined_model:
            #inputs = inputs[1:]
            #x = inputs
            x = inputs[1]
        else:
            x = inputs

        if model_opts.get("input_embedding") and model_opts["input_embedding"]["active"]:

            input_len_except_pose = 0
            if "box" in model_opts["obs_input_type"]:
                input_len_except_pose += 4
            if "speed" in model_opts["obs_input_type"]:
                input_len_except_pose += 1

            features_no_pose = x[:,:,0:input_len_except_pose]
            pose_features = x[:,:,input_len_except_pose:]

            #embedding1 = Dense(20, activation="relu")
            embedding2 = Dense(5, activation="relu")

            x = TimeDistributed(embedding2)(pose_features)

            x = Concatenate()([features_no_pose, x])
        
        if model_opts.get("positional_encoding") and model_opts["positional_encoding"]["active"]:
            x = T2V(1)(x)

        for _ in range(num_transformer_blocks):
            x = TransformerNet.transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

        #x = GlobalAveragePooling1D(data_format="channels_first")(x)
        x = GlobalAveragePooling1D()(x)
        for dim in mlp_units:
            x = Dense(dim, activation="relu")(x)
            x = Dropout(mlp_dropout)(x)

        if include_top:
            #outputs = Dense(n_classes, activation="softmax")(x)
            outputs = Dense(n_classes, activation="sigmoid")(x)
            return Model(inputs, outputs)
        else:
            return x
    
def CombinedModelV1Net(
        weights,
        freeze_conv_layers,
        dropout,
        dense_activation,
        sequence_features_len,
        data_params,
        model_opts,
        mlp_units
    ):
    
    # Build C3D model
    os.makedirs(os.path.dirname(weights), exist_ok=True)
    if not os.path.exists(weights):
        weights_url = 'https://github.com/adamcasson/c3d/releases/download/v0.1/sports1M_weights_tf.h5'
        wget.download(weights_url, weights)
    net_model_1, input_1 = C3DNet(freeze_conv_layers=freeze_conv_layers,
                                  dropout=dropout,
                                  dense_activation=dense_activation,
                                  include_top=False,
                                  weights=weights,
                                  input_layer=True,
                                  sequence_features_len=sequence_features_len,
                            )
    
    # Build basic transformer model
    def concatenate_trajectory_features(inputs, include_img_input=False):
        if len(inputs) > 1:
            inputs = Concatenate(axis=2)(inputs)
        else:
            inputs = inputs[0]
        if include_img_input:
            return [Input(shape=(16, 112, 112, 3)), inputs]
        else:
            return inputs

    # Build base transformer model
    trajectory_inputs = []
    data_sizes = data_params['data_sizes']
    data_types = data_params['data_types']
    core_size = len(data_sizes)

    for i in range(core_size):
        if i == 0:
            continue # first data type consists of image data, not used by the basic transformer
        trajectory_inputs.append(Input(shape=data_sizes[i],
                                    name='input_' + data_types[i]))

    input_2 = concatenate_trajectory_features(trajectory_inputs)

    net_model_2 = TransformerNet.build_model(
        model_opts,
        input_2,
        head_size=256,
        num_heads=4,
        ff_dim=4,
        num_transformer_blocks=4,
        mlp_units=[mlp_units],
        mlp_dropout=0.4,
        dropout=0.25,
        combined_model=False,
        include_top=False
    )

    concat_layer = concatenate([net_model_1, net_model_2], axis=1)
    x = Dense(20, activation='relu', name='fcX')(concat_layer)
    x = Dropout(0.25)(x)

    outputs = Dense(1, activation="sigmoid")(x)

    model = Model([input_1, input_2], outputs)
    return model


def CombinedModelV2Net(
        data_params,
        model_opts,
        mlp_units,
        attention=True
    ):
    
    # Get the C3D model from model file
    c3d_pretrained = load_model(os.path.join("data/models/jaad/C3D/15Jul2023-11h57m22s_EL4", 'model.h5'))
    net_model_1 = Model(c3d_pretrained.input, c3d_pretrained.layers[-2].output) # remove output layer
    net_model_1.summary()

    for layer in net_model_1.layers:
        layer.trainable = False
        layer._name = layer.name + "_c3d"

    # Build basic transformer model
    def concatenate_trajectory_features(inputs, include_img_input=False):
        if len(inputs) > 1:
            inputs = Concatenate(axis=2)(inputs)
        else:
            inputs = inputs[0]
        if include_img_input:
            return [Input(shape=(16, 112, 112, 3)), inputs]
        else:
            return inputs

    trajectory_inputs = []
    data_sizes = data_params['data_sizes']
    data_types = data_params['data_types']
    core_size = len(data_sizes)

    for i in range(core_size):
        if i == 0:
            continue # first data type consists of image data, not used by the basic transformer
        trajectory_inputs.append(Input(shape=data_sizes[i],
                                    name='input_' + data_types[i]))

    input_2 = concatenate_trajectory_features(trajectory_inputs)

    net_model_2 = TransformerNet.build_model(
        model_opts,
        input_2,
        head_size=256,
        num_heads=4,
        ff_dim=4,
        num_transformer_blocks=4,
        mlp_units=[mlp_units],
        mlp_dropout=0.4,
        dropout=0.25,
        combined_model=False,
        include_top=False
    )

    if not attention:
        concat_layer = concatenate([net_model_1.output, net_model_2], axis=1)
        x = Dense(20, activation='relu', name='fcX')(concat_layer)
        x = Dropout(0.25, name='dropout_end_1')(x)
    else:
        x1 = Dense(10, activation='relu', name='fcX1')(net_model_1.output)
        x2 = Dense(10, activation='relu', name='fcX2')(net_model_2)
        concat_layer = concatenate([x1, x2], axis=1)
        x = tf.expand_dims(concat_layer, axis=2) # required by attention layer
        x = CustomAttention()(x)

    # x = tf.squeeze(x, axis=2) add if context is not summed in attention layer
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model([c3d_pretrained.input, input_2], outputs)

    return model


def CombinedModelV3Net(
        data_params,
        model_opts,
        mlp_units,
        attention=False
    ):
    
    # Get the VGG16 Static model from model file
    static_pretrained = load_model(os.path.join("data/models/jaad/Static/29Jul2023-17h14m19s_CM9", 'model.h5'))
    net_model_1 = Model(static_pretrained.input, static_pretrained.layers[-2].output) # remove output layer
    net_model_1.summary()

    for layer in net_model_1.layers:
        layer.trainable = False
        layer._name = layer.name + "_static"

    # Get the C3D model from model file
    c3d_pretrained = load_model(os.path.join("data/models/jaad/C3D/15Jul2023-11h57m22s_EL4", 'model.h5'))
    net_model_2 = Model(c3d_pretrained.input, c3d_pretrained.layers[-2].output) # remove output layer
    net_model_2.summary()

    for layer in net_model_2.layers:
        layer.trainable = False
        layer._name = layer.name + "_c3d"

    # Build basic transformer model
    def concatenate_trajectory_features(inputs, include_img_input=False):
        if len(inputs) > 1:
            inputs = Concatenate(axis=2)(inputs)
        else:
            inputs = inputs[0]
        if include_img_input:
            return [Input(shape=(16, 112, 112, 3)), inputs]
        else:
            return inputs

    trajectory_inputs = []
    data_sizes = data_params['data_sizes']
    data_types = data_params['data_types']
    core_size = len(data_sizes)

    for i in range(core_size):
        if i == 0 or i == 1:
            continue # first two data types consist of image-based data, not used by the basic transformer
        trajectory_inputs.append(Input(shape=data_sizes[i],
                                    name='input_' + data_types[i]))

    input_3 = concatenate_trajectory_features(trajectory_inputs)

    net_model_3 = TransformerNet.build_model(
        model_opts,
        input_3,
        head_size=256,
        num_heads=4,
        ff_dim=4,
        num_transformer_blocks=4,
        mlp_units=[mlp_units],
        mlp_dropout=0.4,
        dropout=0.25,
        combined_model=False,
        include_top=False
    )

    if not attention:
        concat_layer = concatenate([net_model_1.output, net_model_2.output, net_model_3], axis=1)
        x = Dense(20, activation='relu', name='fcX')(concat_layer)
        x = Dropout(0.25, name='dropout_end_1')(x)
    else:
        x1 = Dense(10, activation='relu', name='fcX1')(net_model_1.output)
        x2 = Dense(10, activation='relu', name='fcX2')(net_model_2.output)
        x3 = Dense(10, activation='relu', name='fcX3')(net_model_3)
        concat_layer = concatenate([x1, x2, x3], axis=1)
        x = tf.expand_dims(concat_layer, axis=2) # required by attention layer
        x = CustomAttention()(x)

    # x = tf.squeeze(x, axis=2) add if context is not summed in attention layer
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model([static_pretrained.input, c3d_pretrained.input, input_3], outputs)

    return model

