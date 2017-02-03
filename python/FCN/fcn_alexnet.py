from keras.optimizers import SGD
from convnetskeras.convnets import preprocess_image_batch, convnet
from convnetskeras.imagenet_tool import synset_to_dfs_ids

#im = preprocess_image_batch(['examples/bern.jpg'], color_mode="bgr")
im = preprocess_image_batch(['examples/test.png'], color_mode="bgr")


sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model = convnet('alexnet',weights_path="../../../../models/fully_conv/alexnet_weights.h5", heatmap=True)
#model = convnet('vgg_16',weights_path="../../../../models/fully_conv/vgg16_weights.h5", heatmap=True)

model.compile(optimizer=sgd, loss='mse')

out = model.predict(im)

s = "n02084071"
ids = synset_to_dfs_ids(s)
heatmap = out[0,ids].sum(axis=0)

# Then, we can get the image
import matplotlib.pyplot as plt
plt.imsave("examples/heatmap.png",heatmap)