import cv2

class MeanPreprocessor:
  def __init__(self, rMean, gMean, bMean):
    self.rMean = rMean
    self.gMean = gMean
    self.bMean = bMean
  
  def preprocess(self, image):
    (b, g, r) = cv2.split(image.astype("float32"))
    b -= self.bMean
    g -= self.gMean
    r -= self.rMean

    return cv2.merge([b,g,r])

