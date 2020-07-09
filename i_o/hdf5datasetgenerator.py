from tensorflow.keras.utils import to_categorical
import h5py
import numpy as np

class HDF5DatasetGenerator:
    def __init__(self, dbPath, batch_size, preprocessors=None, aug=None, binarize=True, classes=2):
        self.dbPath = dbPath
        self.batch_size = batch_size
        self.preprocessors= preprocessors
        self.binarize = binarize
        self.aug = aug
        self.classes = classes

        # load file
        self.db = h5py.File(self.dbPath, 'r')
        self.numImages = self.db['labels'].shape[0]
    
    def generator(self, passes=np.inf):
        epochs = 0
        
        while epochs < passes:
        
            for i in np.arange(0,self.numImages, self.batch_size):
                images = self.db['images'][i:i+self.batch_size]
                labels = self.db['labels'][i+i+self.batch_size]

                if self.binarize:
                    labels = to_categorical(labels, self.classes)
                if self.preprocessors is not None:
                    procImages = []
                    for image in images:
                        for p in self.preprocessors:
                            image = p.preprocessor(image)
                            procImages.append(image)
                    images = np.array(procImages)
                
                if self.aug is not None:
                    (images, labels) = next(self.aug.flow(images, labels, batch_size=self.batch_size))
                
                yield (images, labels)
        
            epochs += 1
    

    def close(self):
        self.db.close()