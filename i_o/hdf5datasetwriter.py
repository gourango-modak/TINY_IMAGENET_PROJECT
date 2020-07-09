import h5py

class HDF5DatasetWriter:
  def __init__(self, outputpath, dim, buffer_size=50, dataKey='images'):
    
    # create file
    self.db = h5py.File(outputpath, 'w')
    
    # create dataset
    self.data = self.db.create_dataset(dataKey, dim, dtype='float')
    self.labels = self.db.create_dataset('labels', (dim[0],), dtype='int')

    self.buffer_size = buffer_size
    self.buffer_data = {'data': [], 'labels': []}
    self.index = 0

  
  def add(self, rows, labels):
    self.buffer_data['data'].extend(rows)
    self.buffer_data['labels'].extend(labels)

    if len(self.buffer_data['data']) >= self.buffer_size:
      self.flush()

  
  def flush(self):
    i = self.index + len(self.buffer_data['data'])
    self.data[self.index:i] = self.buffer_data['data']
    self.labels[self.index:i] = self.buffer_data['labels']
    self.index = i
    self.buffer_data = {'data': [], 'labels': []}

  
  def close(self):
    if len(self.buffer_data['data'])>0:
      self.flush()
    self.db.close()