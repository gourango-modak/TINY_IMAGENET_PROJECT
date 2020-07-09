from tensorflow.keras.layers import Conv2D, MaxPool2D, add, AveragePooling2D, Dense, BatchNormalization, Dropout, Activation, Input, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K

class ResNet:
    @staticmethod
    def residual_module(data, K, stride, chanDim, red=False, reg=0.0001, bnEps=2e-5, bnMo=0.9):

        # shortcut brach of the Resnet module should be initialize as input data
        shortcut = data

        # the first block of the residual module 1x1 CONVs
        bn1 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMo)(data)
        act1 = Activation("relu")(bn1)
        conv1 = Conv2D(int(0.25*K), (1,1), use_bias=False, kernel_regularizer=l2(reg))(act1)

        # the second block of the residual module 3x3 CONVs
        bn2 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMo)(conv1)
        act2 = Activation("relu")(bn2)
        conv2 = Conv2D(int(0.25*K), (3,3), use_bias=False, strides=stride, kernel_regularizer=l2(reg), padding="same")(act2)

        # the final block of the residual module 1x1 CONVs
        bn3 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMo)(conv2)
        act3 = Activation("relu")(bn3)
        conv3 = Conv2D(K, (1,1), use_bias=False, kernel_regularizer=l2(reg))(act3)

        if red:
            # if we want to reduce sptial dimension through CONVs
            shortcut = Conv2D(K, (1,1), strides=stride, use_bias=False, kernel_regularizer=l2(reg))(act1)

        # add together final and shortcut CONVs
        model = add([conv3, shortcut])

        return model


    @staticmethod
    def build(width, height, depth, classes, stages, filters, reg=0.0001, bnEps=2e-5, bnMo=0.9, dataset="cifar10"):

        # inputShape
        inputShape = (width, height, depth)
        chanDim = -1

        if K.image_data_format() == "channel_first":
            inputShape = (depth, width, height)
            chanDim = 1

        # set input and BN
        inputs = Input(shape=inputShape)
        x = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMo)(inputs)

        if dataset == "cifar10":
            x = Conv2D(filters[0], (3,3), use_bias=False, kernel_regularizer=l2(reg), padding="same")(x)

        # loop over no of stages
        for i in range(0, len(stages)-1):
            # initialize the stride, then apply residual module to use reduce dimensionality
            stride = (1,1) if i==0 else (2,2)
            x = ResNet.residual_module(x, filters[i+1], stride, chanDim, red=True, bnEps=bnEps, bnMo=bnMo)

            # loop over no of layers in the stage
            for j in range(0, stages[i]-1):
                # apply reset module
                x = ResNet.residual_module(x, filters[i+1], (1,1), chanDim, bnEps, bnMo)
        
        # apply BN => ACT => POOL
        x = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMo)(x)
        x = Activation("relu")(x)
        x = AveragePooling2D((8,8))(x)

        # softmax classifier
        x = Flatten()(x)
        x = Dense(classes, kernel_regularizer=l2(reg))(x)
        x = Activation("relu")(x)

        # create model
        model = Model(inputs=inputs, outputs=x, name="ResNet")

        # return model
        
        return model