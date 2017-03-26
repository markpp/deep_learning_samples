import os, cv2, numpy as np
from small_cnn import CNN


def visualize(path):
    cnn = CNN(path)
    cnn.start_server()

if __name__ == "__main__":
    visualize('weights/small_cnn.h5')