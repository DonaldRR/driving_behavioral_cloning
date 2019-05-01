import tensorflow
import keras
from keras.models import Model, Input
from keras.layers import BatchNormalization, Conv2D, MaxPool2D, Flatten, Dense, Dropout, Lambda, Reshape, Cropping2D, Add, Activation
from keras.optimizers import SGD, Adam
from keras.backend import resize_images
from keras.applications.vgg19 import VGG19
from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50
from keras.applications.mobilenet import MobileNet

# Model creator
def myModel():
    # Input
    input = Input(shape=(160, 320, 3),)
#     input_ = Lambda(lambda x: (x - [106.13, 115.97, 124.96]) / 255.)(input)
    input_ = Cropping2D(((60, 30), (0, 0)))(input)
    input_ = Lambda(lambda x: (x / 127.5) - 1)(input_)


    # Nivdia Net
    conv = Conv2D(3, (5, 5), strides=(2, 2), padding='valid')(input_)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)

    conv = Conv2D(24, (5, 5), strides=(2, 2), padding='valid')(conv)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)

    conv = Conv2D(36, (5, 5), strides=(2, 2), padding='valid')(conv)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)

    conv = Conv2D(48, (3, 3), strides=(2, 2), padding='valid')(conv)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)

    conv = Conv2D(64, (3, 3), strides=(2, 2), padding='valid')(conv)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)

    conv = Flatten()(conv)

    fc = Dense(100)(conv)
    fc = BatchNormalization()(fc)
    fc = Activation('relu')(fc)

    fc = Dense(50)(fc)
    fc = BatchNormalization()(fc)
    fc = Activation('relu')(fc)

    fc = Dense(10)(fc)
    fc = BatchNormalization()(fc)
    fc = Activation('relu')(fc)

    output = Dense(1)(fc)

    
#     # LeNet architecture
#     conv = Conv2D(32, (3, 3), activation='relu')(input_)
# #     conv = BatchNormalization()(conv)
#     conv = Conv2D(32, (3, 3), activation='relu')(conv)
# #     conv = BatchNormalization()(conv)
#     conv = MaxPool2D((2, 2))(conv)
#     conv = Conv2D(64, (3, 3), activation='relu')(conv)
# #     conv = BatchNormalization()(conv)
#     conv = Conv2D(64, (3, 3), activation='relu')(conv)
# #     conv = BatchNormalization()(conv)
#     conv = MaxPool2D((2, 2))(conv)
#     conv = Conv2D(128, (3, 3), activation='relu')(conv)
# #     conv = BatchNormalization()(conv)
#     conv = Conv2D(128, (3, 3), activation='relu')(conv)
# #     conv = BatchNormalization()(conv)
#     conv = MaxPool2D((2, 2))(conv)
    
    
    # Convolutions
    # Resnet of 3 types of building blocks
#     conv1 = Conv2D(32, (5, 5), strides=(2, 2), padding='same', activation='relu')(input_)
#     conv2 = MaxPool2D((2, 2))(conv1)
#     block1_1 = Conv2D(128, (1, 1), padding='same', activation='relu')(conv2)
#     block1_2 = Conv2D(32, (3, 3), padding='same', activation='relu')(block1_1)
#     block1_3 = Conv2D(128, (1, 1), padding='same', activation='relu')(block1_2)
#     block1_4 = Add()([block1_1, block1_3])
#     block2_1 = Conv2D(32, (1, 1), padding='same', activation='relu')(block1_4)
#     block2_2 = Conv2D(32, (3, 3), padding='same', activation='relu')(block2_1)
#     block2_3 = Conv2D(128, (1, 1), padding='same', activation='relu')(block2_2)
#     block2_4 = Add()([block1_4, block2_3])
#     block3_1 = Conv2D(32, (1, 1), padding='same', activation='relu')(block2_4)
#     block3_2 = Conv2D(32, (3, 3), padding='same', activation='relu')(block3_1)
#     block3_3 = Conv2D(128, (1, 1), padding='same', activation='relu')(block3_2)
#     block3_4 = Add()([block2_4, block3_3])
#     blockneck1_1 = MaxPool2D((2, 2))(block3_4)
#     blockneck1_2 = Conv2D(256, (1, 1), padding='same', activation='relu')(blockneck1_1)
#     block4_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(blockneck1_2)
#     block4_2 = Conv2D(256, (1, 1), padding='same', activation='relu')(block4_1)
#     block4_3 = Add()([blockneck1_2, block4_2])
#     block5_1 = Conv2D(64, (1, 1), padding='same', activation='relu')(block4_3)
#     block5_2 = Conv2D(64, (3, 3), padding='same', activation='relu')(block5_1)
#     block5_3 = Conv2D(256, (1, 1), padding='same', activation='relu')(block5_2)
#     block5_4 = Add()([block4_3, block5_3])
#     block6_1 = Conv2D(64, (1, 1), padding='same', activation='relu')(block5_4)
#     block6_2 = Conv2D(64, (3, 3), padding='same', activation='relu')(block6_1)
#     block6_3 = Conv2D(256, (1, 1), padding='same', activation='relu')(block6_2)
#     block6_4 = Add()([block5_4, block6_3])
#     block7_1 = Conv2D(64, (1, 1), padding='same', activation='relu')(block6_4)
#     block7_2 = Conv2D(64, (3, 3), padding='same', activation='relu')(block7_1)
#     block7_3 = Conv2D(256, (1, 1), padding='same', activation='relu')(block7_2)
#     block7_4 = Add()([block6_4, block7_3])
#     blockneck2_1 = MaxPool2D((2, 2))(block7_4)
#     blockneck2_2 = Conv2D(512, (1, 1), padding='same', activation='relu')(blockneck2_1)
#     block8_1 = Conv2D(128, (3, 3), padding='same', activation='relu')(blockneck2_2)
#     block8_2 = Conv2D(512, (1, 1), padding='same', activation='relu')(block8_1)
#     block8_3 = Add()([blockneck2_2, block8_2])
#     block9_1 = Conv2D(128, (1, 1), padding='same', activation='relu')(block8_3)
#     block9_2 = Conv2D(128, (3, 3), padding='same', activation='relu')(block9_1)
#     block9_3 = Conv2D(512, (1, 1), padding='same', activation='relu')(block9_2)
#     block9_4 = Add()([block8_3, block9_3])
#     block10_1 = Conv2D(128, (1, 1), padding='same', activation='relu')(block9_4)
#     block10_2 = Conv2D(128, (3, 3), padding='same', activation='relu')(block10_1)
#     block10_3 = Conv2D(512, (1, 1), padding='same', activation='relu')(block10_2)
#     block10_4 = Add()([block9_4, block10_3])
#     block11_1 = Conv2D(128, (1, 1), padding='same', activation='relu')(block10_4)
#     block11_2 = Conv2D(128, (3, 3), padding='same', activation='relu')(block11_1)
#     block11_3 = Conv2D(512, (1, 1), padding='same', activation='relu')(block11_2)
#     block11_4 = Add()([block10_4, block11_3])
#     block12_1 = Conv2D(128, (1, 1), padding='same', activation='relu')(block11_4)
#     block12_2 = Conv2D(128, (3, 3), padding='same', activation='relu')(block12_1)
#     block12_3 = Conv2D(512, (1, 1), padding='same', activation='relu')(block12_2)
#     block12_4 = Add()([block11_4, block12_3])
#     block13_1 = Conv2D(128, (1, 1), padding='same', activation='relu')(block12_4)
#     block13_2 = Conv2D(128, (3, 3), padding='same', activation='relu')(block13_1)
#     block13_3 = Conv2D(512, (1, 1), padding='same', activation='relu')(block13_2)
#     block13_4 = Add()([block12_4, block13_3])
#     blockneck3_1 = MaxPool2D((2, 2))(block13_4)
#     blockneck3_2 = Conv2D(1024, (1, 1), padding='same', activation='relu')(blockneck3_1)
#     block14_1 = Conv2D(256, (3, 3), padding='same', activation='relu')(blockneck3_2)
#     block14_2 = Conv2D(1024, (1, 1), padding='same', activation='relu')(block14_1)
#     block14_3 = Add()([blockneck3_2, block14_2])
#     block15_1 = Conv2D(256, (1, 1), padding='same', activation='relu')(block14_3)
#     block15_2 = Conv2D(256, (3, 3), padding='same', activation='relu')(block15_1)
#     block15_3 = Conv2D(1024, (1, 1), padding='same', activation='relu')(block15_2)
#     block15_4 = Add()([block14_3, block15_3])
#     block16_1 = Conv2D(256, (1, 1), padding='same', activation='relu')(block15_4)
#     block16_2 = Conv2D(256, (3, 3), padding='same', activation='relu')(block16_1)
#     block16_3 = Conv2D(1024, (1, 1), padding='same', activation='relu')(block16_2)
#     block16_4 = Add()([block15_4, block16_3])
#     blockneck4_1 = MaxPool2D((2, 2))(block16_4)


    # # Denses
    # dense = Flatten()(conv)
    # dense = Dense(32)(dense)
    # dense = Dropout(0.5)(dense)
    # dense = Dense(16)(dense)
    # dense = Dropout(0.5)(dense)
    #
    # # Output
    # output = Dense(1)(dense)

    return Model(inputs=input, outputs=output, name='base')

