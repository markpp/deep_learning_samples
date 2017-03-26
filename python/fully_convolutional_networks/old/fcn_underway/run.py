import os, cv2, numpy as np
#from small_cnn import CNN
from fcn_alex import FCN
from convnetskeras.convnets import preprocess_image_batch


'''
def train():
    cnn = CNN()
    cnn.train()
'''

def predict(path):
    fcn = FCN('model/post_alexnet_weights.h5')

    batch_val = 6
    batch_num = int(120/batch_val)

    for batch in range(batch_num):
        im_list = []
        out_list = []
        for filename in os.listdir(path)[batch*batch_val:batch*batch_val+batch_val]:
            #name, file_extension = filename[:].split('.')
            im_list.append('input/'+ filename[:])
            out_list.append('output/'+ filename[:].split('.')[0])
            #if file_extension == 'png' or file_extension == 'Png' or file_extension == 'jpg' or file_extension == 'Jpeg':
            #    print(filename)
            #    img = cv2.resize(cv2.imread(path+filename), (64, 64)).astype(np.float32)
            #    fcn.predict(img)
        print(im_list[:])
        im = preprocess_image_batch(im_list[:], color_mode="bgr")
        fcn.predict(im, out_list[:])


if __name__ == "__main__":
    #train()
    predict('input/')
