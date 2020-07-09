from config import tiny_imagenet_config as config
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from preprocessing.simplepreprocessor import SimplePreprocessor
from preprocessing.meanpreprocessor import MeanPreprocessor
from preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
import json
from callbacks.epochcheckpoint import EpochCheckpoint
from callbacks.trainingmonitor import TrainingMonitor
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD
from i_o.hdf5datasetgenerator import HDF5DatasetGenerator
from nn.deepergooglenet import DeeperGoogLeNet
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoint", required=True, help="path to output checkpoint directory")
ap.add_argument("-m", "--model", type=str, help="path to *specific* model checkpoint to load")
ap.add_argument("-s", "--start_epoch", type=int, default=0, help="epoch to restart training at")
args = vars(ap.parse_args())

# construct the training image data generator for data augmentation
aug = ImageDataGenerator(rotation_range=18, height_shift_range=0.2, width_shift_range=0.2, zoom_range=0.15,
shear_range=0.15, horizontal_shift=True, fill_mode="nearest")

# load RGB means for training set
means = json.loads(open(config.DATASET_MEAN_PATH,"r").read())

# initialize the image preprocessor
sp = SimplePreprocessor(64, 64)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
iap = ImageToArrayPreprocessor()

# initialize training and validation dataset generator
trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, 64, preprocessors=[sp, mp, iap], aug=aug, classes=config.NUM_CLASSES)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, 64, preprocessors=[sp, mp, iap], aug=aug, classes=config.NUM_CLASSES)

# if there is no specific checkpoint model is supplied then initialize the model and compile

if args["model"] is None:
    print("[INFO] Compiling model.........")
    model = DeeperGoogLeNet.build(64, 64, 3, config.NUM_CLASSES, reg=0.0002)
    opt = SGD(lr=0.01)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
else :
    print("[INFO] loading model {}........".format(args["model"]))
    model = load_model(args["model"])
    print("[INFO] old learning rate: {}".format(K.get_value(model.optimizer.lr)))
    K.set_value(model.optimizer.lr, 1e-5)
    print("[INFO] new learing rate: {}".format(K.get_value(model.optimizer.lr)))

# construct set of all callbacks
callbacks = [
    TrainingMonitor(config.FIG_PATH, config.JSON_PATH),
    EpochCheckpoint(args["checkpoint"], every=5, startAt=args["start_epoch"])
]

# training network
model.fit(
    trainGen.generator(),
    steps_per_epoch = (trainGen.numImages/64),
    validation_data = valGen.generator(),
    validation_steps = (valGen.numImages/64),
    ephochs = 10,
    verbose = 1,
    callbacks = callbacks
)

trainGen.close()
valGen.close()