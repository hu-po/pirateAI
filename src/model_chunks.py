import logging
import keras
from keras import layers, applications

# Define logger
logger = logging.getLogger(__name__)

"""
This file contains chunks of models. Functions that define different
model bases (or encoders), different final dim reduction layers, and
different model heads. These are put in this separate file to prevent cluttering. 
"""


def custom_model(x, params, run_doc):
    """
    Returns the output tensor after going through a custom base
    :param x: input tensor
    :param params: dictionary of params
    :param run_doc: {Ordered dict} run documentation
    :return: output tensor
    """
    # Get parameters from params dictionary, throw errors if not found
    base = params.get('base', None)
    dimreduc = params.get('dim_reduction', None)
    head = params.get('head', None)
    assert base, "No base parameter provided"
    assert dimreduc, "No dim reduction layer parameter provided"
    assert head, "No head parameter provided "
    # Get all possible base functions from globals
    possible = globals().copy()
    possible.update(locals())
    # Base -> Dim Reduction -> Head
    x, run_doc = model_chunk(base, x, params, possible, run_doc)
    x, run_doc = model_chunk(dimreduc, x, params, possible, run_doc)
    x, run_doc = model_chunk(head, x, params, possible, run_doc)
    return x, run_doc


def model_chunk(chunk, x, params, possible, run_doc):
    """
    Runs input through a model chunk (function)
    :param chunk: (string) name of model chunk
    :param x: input tensor
    :param params: {dict} hyperparams (sub-selection)
    :param possible: {dict} local function names
    :param run_doc: {Ordered dict} run documentation
    :return: output tensor, run documentation
    :raises ValueError: could not find parameter
    """
    func = possible.get(chunk)
    sub_param = params.get(chunk + ' params', None)
    if not func:
        raise ValueError('Could not find %s layer' % chunk)
    x = func(x, sub_param)
    run_doc[chunk] = sub_param
    return x, run_doc


def _big_base(func):
    """
    Decorator for big bases (xception, inception, etc)
    :raise ValueError: missing parameters
    """

    def wrapper(x, params):
        assert params, 'Model chunk needs params'
        pre_trained = params.get('pre_trained', None)
        trainable = params.get('trainable', None)
        input_shape = params.get('input_shape', None)
        if any(p is None for p in [pre_trained, trainable, input_shape]):
            raise ValueError('xception missing argument')
        # Option for pre-trained weights from imagenet
        weights = 'imagenet' if pre_trained else None
        base = func(weights, input_shape)
        base.trainable = trainable  # optionally freeze the base
        return base(x)

    return wrapper


@_big_base
def xception(weights, input_shape):
    """
    Chollet's Xception architechture, supposedly better than InceptionV4
    :param weights: (bool)
    :param input_shape: tuple(3)
    :return: base
    """
    base = applications.xception.Xception(weights=weights,
                                          input_shape=input_shape,
                                          include_top=False)
    return base


@_big_base
def inception_res_v2(weights, input_shape):
    """
    Google's Inception Resnet V2
    :param weights: (bool)
    :param input_shape: tuple(3)
    :return: base
    """
    # Option for pre-trained weights from imagenet
    base = applications.inception_resnet_v2.InceptionResNetV2(weights=weights,
                                                              input_shape=input_shape,
                                                              include_top=False)
    return base


@_big_base
def res_net_50(weights, input_shape):
    """
    The Resnet 50. Residual Connections.
    :param weights: (bool)
    :param input_shape: tuple(3)
    :return: base
    """
    # Option for pre-trained weights from imagenet
    base = applications.resnet50.ResNet50(weights=weights,
                                          input_shape=input_shape,
                                          include_top=False)
    return base


def a3c(x, params):
    """
    Feed forward model used in a3c paper
    :param x: input tensor
    :param params: {dict} hyperparams (sub-selection)
    :return: output tensor
    :raises ValueError: could not find parameter
    """
    x = layers.Conv2D(filters=16, kernel_size=8, strides=4, activation='relu')(x)
    x = layers.Conv2D(filters=32, kernel_size=4, strides=2, activation='relu')(x)
    return x


def a3c_sepconv(x, params):
    """
    Feed forward model used in a3c paper but with seperable convolutions
    :param x: input tensor
    :param params: {dict} hyperparams (sub-selection)
    :return: output tensor
    :raises ValueError: could not find parameter
    """
    x = layers.SeparableConv2D(filters=16, kernel_size=8, strides=4, activation='relu')(x)
    x = layers.SeparableConv2D(filters=32, kernel_size=4, strides=2, activation='relu')(x)
    return x


def minires(x, params):
    """
    Small net with residual connections
    :param x: input tensor
    :param params: {dict} hyperparams (sub-selection)
    :return: output tensor
    :raises ValueError: could not find parameter
    """
    x_a = layers.Conv2D(16, 4, activation='relu', padding='same')(x)
    x_b = layers.Conv2D(16, 8, activation='relu', padding='same')(x)
    x_1 = keras.layers.concatenate([x_a, x_b])
    x_c = layers.Conv2D(32, 4, activation='relu', padding='same')(x_1)
    x_d = layers.Conv2D(32, 8, activation='relu', padding='same')(x_1)
    x = keras.layers.concatenate([x_c, x_d, x_1])
    return x


def tall_kernel(x, params):
    """
    Small net with residual connections and tall kernels
    :param x: input tensor
    :param params: {dict} hyperparams (sub-selection)
    :return: output tensor
    :raises ValueError: could not find parameter
    """
    x_a = layers.Conv2D(16, kernel_size=(2, 8), activation='relu', padding='same')(x)
    x_b = layers.Conv2D(16, kernel_size=(8, 2), activation='relu', padding='same')(x)
    x_1 = keras.layers.concatenate([x_a, x_b])
    x_c = layers.Conv2D(32, kernel_size=(2, 8), activation='relu', padding='same')(x_1)
    x_d = layers.Conv2D(32, kernel_size=(8, 2), activation='relu', padding='same')(x_1)
    x = keras.layers.concatenate([x_c, x_d, x_1])
    return x


def simpleconv(x, params):
    """
    Simple CNN base.
    :param x: input tensor
    :param params: {dict} hyperparams (sub-selection)
    :return: output tensor
    :raises ValueError: could not find parameter
    """
    x = layers.Conv2D(16, 3, activation='relu')(x)
    x = layers.Conv2D(32, 3, activation='relu')(x)
    return x


def fc(x, params):
    """
    Fully connected net head
    :param x: input tensor
    :param params: {dict} hyperparams (sub-selection)
    :return: output tensor
    """
    assert params, 'Model chunk needs params'
    # Head consists of a couple dense layers with dropout
    for i in range(0, len(params["dense_layers"])):
        x = layers.Dense(params["dense_layers"][i], activation=params["dense_activations"])(x)
        x = layers.Dropout(params['dropout_percentage'])(x)
    return x


def global_average(x, params):
    """
    Global average pooling
    :param x: input tensor
    :param params: {dict} hyperparams (sub-selection)
    :return: output tensor
    """
    return layers.GlobalAveragePooling2D()(x)


def global_max(x, params):
    """
    Global max pooling
    :param x: input tensor
    :param params: {dict} hyperparams (sub-selection)
    :return: output tensor
    """
    return layers.GlobalMaxPool2D()(x)


def flatten(x, params):
    """
    Plain ol' 2D flatten
    :param x: input tensor
    :param params: {dict} hyperparams (sub-selection)
    :return: output tensor
    """
    return layers.Flatten()(x)
