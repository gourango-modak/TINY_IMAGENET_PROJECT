from nn.deepergooglenet import DeeperGoogLeNet

model = DeeperGoogLeNet.build(64, 64, 3, 200)
model.summary()