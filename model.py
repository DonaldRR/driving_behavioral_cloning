import tensorflow
import keras
from keras.models import Model, Input
from keras.layers import BatchNormalization, Conv2D, MaxPool2D, Flatten, Dense, Dropout, Lambda, Reshape, Cropping2D
from keras.optimizers import SGD, Adam
from keras.backend import resize_images
from keras.applications.vgg19 import VGG19
from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50

# Model creator
def model():
    # Input
    input = Input(shape=(224, 224, 3), )

    # Crop and resize input
    #     input_ = Cropping2D(((61, 0), (0, 0)))(input)
    #     input_ = Lambda(lambda x: resize_images(x, 0.5, 0.5, data_format="channels_last"))(input)

    vgg19_base = VGG19(include_top=False)
    for i in range(len(vgg19_base.layers)):
        vgg19_base.layers[i].trainable = False
    for i in range(10):
        vgg19_base.layers.pop()

        # Convolutions
    #     conv = BatchNormalization()(input_)
    #     conv = Conv2D(16, (3, 3), activation='relu')(conv)
    #     conv = Conv2D(16, (3, 3), activation='relu')(conv)
    #     conv = MaxPool2D((2, 2))(conv)
    #     conv = Conv2D(32, (3, 3), activation='relu')(conv)
    #     conv = Conv2D(32, (3, 3), activation='relu')(conv)
    #     conv = MaxPool2D((2, 2))(conv)

    vgg = vgg19_base(input)

    # Denses
    dense = Flatten()(vgg)
    dense = Dense(32)(dense)
    dense = Dropout(0.5)(dense)
    dense = Dense(16)(dense)
    dense = Dropout(0.5)(dense)

    # Output
    output = Dense(1)(dense)

    return Model(inputs=input, outputs=output, name='base')

