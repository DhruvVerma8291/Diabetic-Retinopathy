import tensorflow as tf
from tensorflow import keras
from keras import layers


def activation_block(x):
    x = layers.Activation("gelu")(x)
    return layers.BatchNormalization()(x)


def conv_stem(x, filters, patch_size):
    x = layers.Conv2D(filters, kernel_size=patch_size, strides=patch_size)(x)
    return activation_block(x)


def conv_mixer_block(x, filters, kernel_size):
    x0 = x

    x = layers.DepthwiseConv2D(kernel_size=kernel_size, padding="same")(x)
    x = layers.Add()([activation_block(x), x0])

    x = layers.Conv2D(filters, kernel_size=1)(x)
    x = activation_block(x)

    return x


def build_convmixer(
        image_size=224,
        filters=256,
        depth=8,
        kernel_size=5,
        patch_size=2,
        num_classes=5):

    inputs = keras.Input((image_size, image_size, 3))

    x = layers.Rescaling(1./255)(inputs)

    x = conv_stem(x, filters, patch_size)

    for _ in range(depth):
        x = conv_mixer_block(x, filters, kernel_size)

    x = layers.GlobalAvgPool2D()(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)

    return model
