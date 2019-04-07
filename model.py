import tensorflow
import keras
from keras.models import Model, Input
from keras.layers import BatchNormalization, Conv2D, MaxPool2D, Flatten, Dense, Dropout, Lambda, Reshape, Cropping2D, Add
from keras.optimizers import SGD, Adam
from keras.backend import resize_images
from keras.applications.vgg19 import VGG19
from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50

# Model creator
def model():
    # Input
    input = Input(shape=(160, 320, 3), )

    # Crop and resize input
    input_ = Cropping2D(((60, 0), (0, 0)))(input)
    input_ = BatchNormalization()(input_)

    # Convolutions
    # (100, 320, 3)
    conv1 = Conv2D(32, (5, 5), strides=(2, 2), padding='same', activation='relu')(input_)
    # (50, 160, 32)
    conv2 = MaxPool2D((2, 2))(conv1)
    # (25, 80, 32)
    block1_1 = Conv2D(128, (1, 1), padding='same', activation='relu')(conv2)
    # (25, 80, 128)
    block1_2 = Conv2D(32, (3, 3), padding='same', activation='relu')(block1_1)
    # (25, 80, 32)
    block1_3 = Conv2D(128, (1, 1), padding='same', activation='relu')(block1_2)
    # (25, 80, 128)
    block1_4 = Add()([block1_1, block1_3])
    # (25, 80, 128)
    block2_1 = Conv2D(32, (1, 1), padding='same', activation='relu')(block1_4)
    # (25, 80, 32)
    block2_2 = Conv2D(32, (3, 3), padding='same', activation='relu')(block2_1)
    # (25, 80, 32)
    block2_3 = Conv2D(128, (1, 1), padding='same', activation='relu')(block2_2)
    # (25, 80, 128)
    block2_4 = Add()([block1_4, block2_3])
    # (25, 80, 128)
    block3_1 = Conv2D(32, (1, 1), padding='same', activation='relu')(block2_4)
    # (25, 80, 32)
    block3_2 = Conv2D(32, (3, 3), padding='same', activation='relu')(block3_1)
    # (25, 80, 32)
    block3_3 = Conv2D(128, (1, 1), padding='same', activation='relu')(block3_2)
    # (25, 80, 128)
    block3_4 = Add()([block2_4, block3_3])
    # (25, 80, 128)
    blockneck1_1 = MaxPool2D((2, 2))(block3_4)
    # (12, 40, 128)
    blockneck1_2 = Conv2D(256, (1, 1), padding='same', activation='relu')(blockneck1_1)
    # (12, 40, 256)
    block4_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(blockneck1_2)
    # (12, 40, 64)
    block4_2 = Conv2D(256, (1, 1), padding='same', activation='relu')(block4_1)
    # (12, 40, 256)
    block4_3 = Add()([blockneck1_2, block4_2])
    # (12, 40, 256)
    block5_1 = Conv2D(64, (1, 1), padding='same', activation='relu')(block4_3)
    # (12, 40, 64)
    block5_2 = Conv2D(64, (3, 3), padding='same', activation='relu')(block5_1)
    # (12, 40, 64)
    block5_3 = Conv2D(256, (1, 1), padding='same', activation='relu')(block5_2)
    # (12, 40, 256)
    block5_4 = Add()([block4_3, block5_3])
    # (12, 40, 256)
    block6_1 = Conv2D(64, (1, 1), padding='same', activation='relu')(block5_4)
    # (12, 40, 64)
    block6_2 = Conv2D(64, (3, 3), padding='same', activation='relu')(block6_1)
    # (12, 40, 64)
    block6_3 = Conv2D(256, (1, 1), padding='same', activation='relu')(block6_2)
    # (12, 40, 256)
    block6_4 = Add()([block5_4, block6_3])
    # (12, 40, 256)
    blockneck2_1 = MaxPool2D((2, 2))(block6_4)
    # (6, 20, 256)

    # Denses
    dense = Flatten()(blockneck2_1)
    dense = Dense(32)(dense)
    dense = Dropout(0.5)(dense)
    dense = Dense(16)(dense)
    dense = Dropout(0.5)(dense)

    # Output
    output = Dense(1)(dense)

    return Model(inputs=input, outputs=output, name='base')

