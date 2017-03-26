import os
import cv2
import json
import numpy as np
import finetune_vgg
from predict_vgg import fcn_vgg
from convnetskeras.convnets import preprocess_image_batch


def train(config_path):
    """Train CNN.

    The CNN is trained using a configuration described
    in the .json file under the configs/ dir.

    Args:
    """
    finetune_vgg.train(config_path)


def predict(config_path, img_path):
    """Predict using trained FCN.

    The FCN is loaded and the images located in 'img_path' are divided
    into batches before being being labelled by the FCN.

    Args:
    """
    config = json.load(open(config_path))

    fcn = fcn_vgg(config['output_weight_path'])

    batch_size = 1  # 6
    test_sample = 2  # 120
    batch_num = int(test_sample/batch_size)

    for batch in range(batch_num):
        print("Batch number: {}".format(batch))
        im_list = []
        out_list = []
        for filename in os.listdir(img_path)[batch*batch_size:batch*batch_size+batch_size]:
            #name, file_extension = filename[:].split('.')
            im_list.append('../input/'+ filename[:])
            out_list.append('../output/'+ filename[:].split('.')[0])
            #if file_extension == 'png' or file_extension == 'Png' or file_extension == 'jpg' or file_extension == 'Jpeg':
            #    print(filename)
            #    img = cv2.resize(cv2.imread(img_path+filename), (64, 64)).astype(np.float32)
            #    fcn.predict(img)
        print(im_list[:])
        im = preprocess_image_batch(im_list[:], color_mode="bgr")
        fcn.predict(im, out_list[:])


if __name__ == "__main__":
    config_path = 'configs/mac_vgg16_organs.json'
    #train(config_path)
    predict(config_path, '../input/')
