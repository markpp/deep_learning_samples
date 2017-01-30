import os
import cv2
import numpy as np
#from train_top import CNN
from vgg16 import CNN


def train_top():
    cnn = CNN()
    #cnn.save_bottlebeck_features()
    cnn.train_top_model()

def finetune_top():
    cnn = CNN()
    #cnn.save_bottlebeck_features()
    cnn.fine_tune()

def predict(path):
    cnn = CNN()
    #model = cnn.load_weights()
    model = cnn.fine_tune()

    for filename in os.listdir(path):
        name, file_extension = filename[:].split('.')
        if file_extension == 'png' or file_extension == 'Png' or file_extension == 'jpg' or file_extension == 'Jpeg':
            print(filename)
            img = cv2.resize(cv2.imread(path+filename), (224, 224)).astype(np.float32)
            img = img.transpose((2, 0, 1))
            img = np.expand_dims(img, axis=0)
            out = model.predict(img)
            print('predicted label: {}, probs: {}'.format(np.argmax(out), out))


if __name__ == "__main__":
    # only instantiating the convolutional part of the model. This model is then run with our training and validation data once, recording the output (the "bottleneck features" from th VGG16 model: the last activation maps before the fully-connected layers) in two numpy arrays. Then we will train a small fully-connected model on top of the stored features.
    train_top()

    # Finetune top layers
    #finetune_top()

    #
    #predict('../../data/')
