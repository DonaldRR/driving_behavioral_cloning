import tensorflow
import keras
from keras.models import Model, Input
from keras.layers import BatchNormalization, Conv2D, MaxPool2D, Flatten, Dense, Dropout, Lambda, Reshape
from keras.optimizers import SGD, Adam
from keras.backend import resize_images

# Model creator
def model():
    # Input
    input = Input(shape=(160, 320, 3), )
    print(input.get_shape())
    # Crop and resize input
    #     input_ = Lambda(lambda x: x[:, 60:, :, :])(input)
    # input_ = Lambda(lambda x: resize_images(x, 0.5, 0.5, data_format="channels_last"))(input)

    # Convolutions
    conv = BatchNormalization()(input)
    conv = Conv2D(8, (3, 3), activation='relu')(conv)
    conv = MaxPool2D((2, 2))(conv)
    conv = Conv2D(16, (3, 3), activation='relu')(conv)
    conv = MaxPool2D((2, 2))(conv)

    # Denses
    dense = Flatten()(conv)
    dense = Dense(32)(dense)
    dense = Dropout(0.5)(dense)
    dense = Dense(16)(dense)

    # Output
    output = Dense(1)(dense)

    return Model(inputs=input, outputs=output, name='base')

