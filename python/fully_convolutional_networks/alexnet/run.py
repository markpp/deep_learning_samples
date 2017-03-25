import os
import cv2
import numpy as np
from finetune_alex import tune_alex
from predict_alex import fcn_alex
from convnetskeras.convnets import preprocess_image_batch


def train():
    """Train CNN.

    The CNN is trained using a configuration described
    in the .json file under the configs/ dir.

    Args:
    """
    tune_alex.train('configs/p50_alex_setup_organs_small.json')


def predict(img_path):
    """Predict using trained FCN.

    The FCN is loaded and the images located in 'img_path' are divided
    into batches before being being labelled by the FCN.

    Args:
    """
    fcn = fcn_alex('model/post_alexnet_weights.h5')

    batch_val = 6
    batch_num = int(120/batch_val)

    for batch in range(batch_num):
        im_list = []
        out_list = []
        for filename in os.listdir(img_path)[batch*batch_val:batch*batch_val+batch_val]:
            #name, file_extension = filename[:].split('.')
            im_list.append('input/'+ filename[:])
            out_list.append('output/'+ filename[:].split('.')[0])
            #if file_extension == 'png' or file_extension == 'Png' or file_extension == 'jpg' or file_extension == 'Jpeg':
            #    print(filename)
            #    img = cv2.resize(cv2.imread(img_path+filename), (64, 64)).astype(np.float32)
            #    fcn.predict(img)
        print(im_list[:])
        im = preprocess_image_batch(im_list[:], color_mode="bgr")
        fcn.predict(im, out_list[:])


if __name__ == "__main__":
    train()
    #predict('input/')
