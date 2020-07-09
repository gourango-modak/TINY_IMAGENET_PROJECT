from tensorflow.keras.layers import Dense, Activation, MaxPool2D, Conv2D, BatchNormalization, Dropout, AveragePooling2D, Flatten, Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K

class DeeperGoogLeNet:
    @staticmethod
    def conv_model(model, K, kX, kY, stride, chanDim, padding="same", reg=0.001, name=None):
        # initialize CONV , BN, and Activation names
        (convName, bnName, actName) = (None, None, None)
        if name is not None:
            convName = name + "_conv"
            bnName = name + "_bn"
            actName = name + "_act"
        model = Conv2D(K, (kX, kY), padding=padding, strides=stride, kernel_regularizer=l2(reg), name=convName)(model)
        model = BatchNormalization(axis=chanDim, name=bnName)(model)
        model = Activation('relu', name=actName)(model)

        return model

    @staticmethod
    def inception_module(model, num1x1, num3x3Reduce, num3x3, num5x5Reduce, num5x5, num1x1Proj, chanDim, stage, reg=0.001):
        # define the first branch of the inception module which consists of 1x1 conv
        first = DeeperGoogLeNet.conv_model(model, num1x1, 1, 1, (1, 1), chanDim, reg=reg, name=stage+"_first")
        # define the second branch of the inception module which consists of 1x1 and 3x3 conv
        second = DeeperGoogLeNet.conv_model(model, num3x3Reduce, 1, 1, (1, 1), chanDim, reg=reg, name=stage+"_second1")
        second = DeeperGoogLeNet.conv_model(second, num3x3, 3, 3, (1, 1), chanDim, reg=reg, name=stage+"_second2")
        # define the third branch of the inception module which consists of 1x1 and 5x5 conv
        third = DeeperGoogLeNet.conv_model(model, num5x5Reduce, 1, 1, (1, 1), chanDim, reg=reg, name=stage+"_third1")
        third = DeeperGoogLeNet.conv_model(third, num5x5, 5, 5, (1, 1), chanDim, reg=reg, name=stage+"_third2")
        # define the forth branch of the inception module which consists of 1x1 and MaxPooling conv
        fourth = MaxPool2D((3,3), strides=(1,1), padding="same", name=stage+"_pool")(model)
        fourth = DeeperGoogLeNet.conv_model(fourth, num1x1Proj, 1, 1, (1, 1), chanDim, reg=reg, name=stage+"_fourth")
        # concate channel dimension
        model = concatenate([first, second, third, fourth], name=stage+"_mixed", axis=chanDim)

        return model

    @staticmethod
    def build(width, height, depth, classes, reg=0.0001):
        # inputShape
        inputShape = (width, height, depth)
        chanDim = -1

        # if keras backend use depth in first then depth order will be change
        if K.image_data_format() == "channel_first":
            inputShape = (depth, width, height)
            chanDim = 1
        
        # define model input layer, followed by conv => Pooling => 2*conv => Pooling
        inputs = Input(shape=inputShape)
        model = DeeperGoogLeNet.conv_model(inputs, 64, 5, 5, (1, 1), chanDim, reg=reg, name="block_1")
        model = MaxPool2D((3, 3), (2, 2), name="pool_1", padding="Same")(model)
        
        model = DeeperGoogLeNet.conv_model(model, 64, 1, 1, (1, 1), chanDim, reg=reg, name="block_2")
        model = DeeperGoogLeNet.conv_model(model, 192, 3, 3, (1, 1), chanDim, reg=reg, name="block_3")
        model = MaxPool2D((3, 3), (2, 2), name="pool_2", padding="Same")(model)

        # apply two inception module followed by pool
        model = DeeperGoogLeNet.inception_module(model, 64, 96, 128, 16, 32, 32, chanDim, "3a", reg)
        model = DeeperGoogLeNet.inception_module(model, 128, 128, 192, 32, 96, 64, chanDim, "3b", reg)
        model = MaxPool2D((3, 3), (2, 2), name="pool_3", padding="Same")(model)


        # apply five inception module followed by pool
        model = DeeperGoogLeNet.inception_module(model, 192, 96, 208, 16, 48, 64, chanDim, "4a", reg)
        model = DeeperGoogLeNet.inception_module(model, 160, 112, 224, 24, 64, 64, chanDim, "4b", reg)
        model = DeeperGoogLeNet.inception_module(model, 128, 128, 256, 24, 64, 64, chanDim, "4c", reg)
        model = DeeperGoogLeNet.inception_module(model, 112, 144, 288, 32, 64, 64, chanDim, "4d", reg)
        model = DeeperGoogLeNet.inception_module(model, 256, 160, 320, 32, 128, 128, chanDim, "4e", reg)
        model = MaxPool2D((3, 3), (2, 2), name="pool_4", padding="Same")(model)

        # apply Average Pool layer followd by Dropout
        model = AveragePooling2D((4, 4), name="pool5")(model)
        model = Dropout(0.4, name="dropout")(model)

        # softmax classifier
        model = Flatten(name="flatten")(model)
        model = Dense(classes, kernel_regularizer=l2(reg), name="labels")(model)
        model = Activation("softmax", name="softmax")(model)

        # create model
        model = Model(inputs=inputs, outputs=model, name="GoogLeNet")

        return model