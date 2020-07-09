from tensorflow.keras.callbacks import BaseLogger
import json
import os
import numpy as np
import matplotlib.pyplot as plt

class TrainingMonitor(BaseLogger):
  def __init__(self, figPath, jsonPath=None, startAt=0):
    super(TrainingMonitor, self).__init__()
    self.figPath = figPath
    self.jsonPath = jsonPath
    self.startAt = startAt

  def on_train_begin(self, logs={}):
    self.H = {}
    if self.jsonPath is not None:
      if os.path.exists(self.jsonPath):
        self.H = json.loads(open(self.jsonPath).read())
        if self.startAt>0:
          for k in self.H.keys():
            self.H[k] = self.H[k][:self.startAt]

  
  def on_epoch_end(self, epoch, logs={}):
    for (k,v) in logs.items():
      l = self.H.get(k,[])
      l.append(float(v)) # Here by default float is float32 which is not support to serialize
      self.H[k] = l
    if self.jsonPath is not None:
      f = open(self.jsonPath, 'w')
      f.write(json.dumps(self.H))
      f.close()
    if len(self.H['loss'])>3:
      ephochs = np.arange(0, len(self.H['loss']))
      plt.figure()
      plt.plot(ephochs, self.H['loss'], label='loss')
      plt.plot(ephochs, self.H['val_loss'], label='validation_loss')
      plt.plot(ephochs, self.H['accuracy'], label='accuracy')
      plt.plot(ephochs, self.H['val_accuracy'], label='validation_accuracy')
      plt.title('Trainning Loss and Accuracy')
      plt.xlabel("No of Epochs")
      plt.ylabel("Loss/Accuracy")
      plt.legend()
      plt.savefig(self.figPath)
      plt.close()