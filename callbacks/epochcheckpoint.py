from tensorflow.keras.callbacks import BaseLogger

class EpochCheckpoint(BaseLogger):
  def __init__(self, checkpointPath, every=5, startAt=0):
    super(EpochCheckpoint, self).__init__()
    self.checkpointPath = checkpointPath
    self.every = every
    self.startAt = startAt

  def on_epoch_end(self, epoch, logs={}):
      if self.startAt>0:
          epoch += self.startAt
      if epoch>0 and (epoch%self.every == 0):
          self.model.save("model_epoch_{}.hdf5".format(epoch))