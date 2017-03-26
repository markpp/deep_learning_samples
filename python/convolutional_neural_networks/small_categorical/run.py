import os, cv2, numpy as np
from small_cnn import CNN


def train():
    cnn = CNN()
    cnn.train()


def predict(path):
    cnn = CNN('weights/small_cnn.h5')

    for filename in os.listdir(path):
        name, file_extension = filename[:].split('.')
        if file_extension == 'png' or file_extension == 'Png' or file_extension == 'jpg' or file_extension == 'Jpeg':
            print filename
            img = cv2.resize(cv2.imread(path+filename), (224, 224)).astype(np.float32)
            cnn.predict(img)


if __name__ == "__main__":
    train()
    predict('../../data/')
