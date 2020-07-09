from config import tiny_imagenet_config as config
from tensorflow.keras.models import load_model
from i_o.hdf5datasetgenerator import HDF5DatasetGenerator
from preprocessing.simplepreprocessor import SimplePreprocessor
from preprocessing.meanpreprocessor import MeanPreprocessor
from preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from utils.ranked import rank5_accuracy
import json


# load RGB channel mean values
means = json.loads(open(config.DATASET_MEAN_PATH).read())

# initialization the image preprocessors
sp = SimplePreprocessor(64, 64)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
iap = ImageToArrayPreprocessor()

# initialize testing dataset generator
testGen = HDF5DatasetGenerator(config.TEST_HDF5, 64, preprocessors=[sp, mp, iap], classes=config.NUM_CLASSES)

# load the pre-trained network
print("[INFO] loading model........")
model = load_model(config.MODEL_PATH)

# make prediction on test data
print("[INFO] predicting on test data........")
preds = model.predict_generator(testGen.generator(), steps=(testGen.numImages/64))

# compute rank-1 and rank-5 accuracies
(rank1, rank5) = rank5_accuracy(preds, testGen.db["labels"].generator())
print("[INFO] Rank - 1 : {:.2f}%".format(rank1*100))
print("[INFO] Rank - 5 : {:.2f}%".format(rank5*100))

testGen.close()