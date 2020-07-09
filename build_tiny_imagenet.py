from config import tiny_imagenet_config as config
import h5py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from i_o.hdf5datasetwriter import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
import json
import cv2
import os 

# grab the path to the images
trainPaths = list(paths.list_images(config.TRAIN_IMAGE_PATH))
trainLabels = [p.split(os.path.sep)[-3] for p in trainPaths]

# encode labels to ont-hot
lb = LabelEncoder()
trainLabels = lb.fit_transform(trainLabels)


# train and test split
(trainPaths, testPaths, trainLabels, testLabels) = train_test_split(trainPaths, 
trainLabels, test_size=config.NO_TEST_IMAGES, stratify=trainLabels, random_state=42)

# build validation data
valPaths = []
valLabels = []
with open(config.VAL_MAPPINGS,'r') as f:
    for file in f:
        valPaths.append(os.path.sep.join([config.VAL_IMAGE_PATH, file.split('\t')[0]]))
        valLabels.append(file.split('\t')[1])

valLabels = lb.fit_transform(valLabels)

# define dictionary to store training, testing and validation paths, labels and hdf5 files
datasets = [
    ("train", trainPaths, trainLabels, config.TRAIN_HDF5),
    ("test", testPaths, testLabels, config.TEST_HDF5),
    ("val", valPaths, valLabels, config.VAL_HDF5)
]

# initialize the list of RGB channel averages
(R, G, B) = ([], [], [])

for (dType, paths, labels, output) in datasets:

    # create HDF5 file
    print('[INFO] building {}......',format(output))
    write = HDF5DatasetWriter(output, (len(paths), 64, 64, 3))

    # build a progressbar
    widget = ["Building Dataset: ", progressbar.Percentage()," ", progressbar.Bar()," ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(paths), widgets=widget).start()

    # load each images from paths
    for (i,(image, label)) in enumerate(zip(paths, labels)):
        image = cv2.imread(image)

        if dType == "train":
            (b, g, r) = cv2.mean(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)
        write.add([image], [label])
        pbar.update(i)
    write.close()
    pbar.finish()

# store RGB mean values
print('[INFO] serializing means....')
f = open(config.DATASET_MEAN_PATH, 'w')
means = {"R": np.mean(R), "G": np.mean(G), "B": np.mean(B)}
f.write(json.dumps(means))
f.close()