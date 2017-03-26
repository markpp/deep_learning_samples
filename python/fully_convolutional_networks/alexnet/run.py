import os
import cv2
import json

with open("configs/mac_alex_organs.json") as config_file:
    json_config = json.load(config_file)
if json_config['gpu'] == 1:
    os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"
else:
    os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"

import numpy as np
import finetune_alex
from predict_alex import fcn_alex
from convnetskeras.convnets import preprocess_image_batch


def train(config_path):
    """Train CNN.

    The CNN is trained using a configuration described
    in the .json file under the configs/ dir.

    Args:
    """
    finetune_alex.train(config_path)


def predict(config_path, img_path):
    """Predict using trained FCN.

    The FCN is loaded and the images located in 'img_path' are divided
    into batches before being being labelled by the FCN.

    Args:
    """
    config = json.load(open(config_path))

    fcn = fcn_alex(config['output_weight_path'])

    batch_val = 6
    batch_num = int(120/batch_val)

    for batch in range(batch_num):
        im_list = []
        out_list = []
        for filename in os.listdir(img_path)[batch*batch_val:batch*batch_val+batch_val]:
            #name, file_extension = filename[:].split('.')
            im_list.append('../input/'+ filename[:])
            out_list.append('../output/'+ filename[:].split('.')[0])
            #if file_extension == 'png' or file_extension == 'Png' or file_extension == 'jpg' or file_extension == 'Jpeg':
            #    print(filename)
            #    img = cv2.resize(cv2.imread(img_path+filename), (64, 64)).astype(np.float32)
            #    fcn.predict(img)
        print(im_list[:])
        #im = preprocess_image_batch(im_list[:], color_mode="bgr")
        im = preprocess_image_batch(im_list[:], color_mode="rgb")

        fcn.predict(im, out_list[:])


if __name__ == "__main__":
    config_path = 'configs/mac_alex_organs.json'
    #train(config_path)
    predict(config_path, '../input/')
