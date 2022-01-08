from tensorflow.keras import Input, Sequential, Model
from tensorflow.keras.layers import MaxPooling2D, Conv2D, Flatten, Dense, Lambda, GlobalAveragePooling2D, \
    Activation
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K


def get_siamese_model(input_shape):
    """
        Model architecture
    """

    left_input = Input(input_shape)
    right_input = Input(input_shape)
    # build convnet to use in each siamese 'leg'
    convnet = Sequential()
    convnet.add(Conv2D(32, (10, 10), activation='relu', input_shape=input_shape))
    convnet.add(MaxPooling2D())
    convnet.add(Conv2D(64, (7, 7), activation='relu'))
    convnet.add(MaxPooling2D())
    convnet.add(Conv2D(64, (4, 4), activation='relu'))
    convnet.add(MaxPooling2D())
    convnet.add(Conv2D(128, (4, 4), activation='relu'))
    convnet.add(MaxPooling2D())
    convnet.add(GlobalAveragePooling2D())
    convnet.add(Activation('sigmoid'))
    # encode each of the two inputs into a vector with the convnet
    encoded_l = convnet(left_input)
    encoded_r = convnet(right_input)
    # merge two encoded inputs with the l1 distance between them
    both = Lambda(lambda x: K.abs(x[0] - x[1]))([encoded_l, encoded_r])
    prediction = Dense(1, activation='sigmoid')(both)
    siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)

    # return the model
    return siamese_net
