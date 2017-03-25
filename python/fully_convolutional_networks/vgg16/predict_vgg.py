import cv2
import numpy as np
from keras.optimizers import SGD
from convnetskeras.convnets import preprocess_image_batch, convnet


class fcn_vgg:
    def __init__(self, weights_path=None):
        self.img_width, self.img_height = 64, 64  # dimensions of our images.
        self.out_width, self.out_height = 150, 350  # dimensions of our images.
        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        print("loading vgg weights from: {}".format(weights_path))
        self.model = convnet('vgg_16',weights_path=weights_path, heatmap=True)
        #model = convnet('vgg_16', weights_path="post_vgg16_weights.h5", heatmap=True)

        self.model.compile(optimizer=sgd, loss='mse')


    def predict(self, im, out_list):
        print(im.shape)
        #im[:,:,0] -= 103.939
        #im[:,:,1] -= 116.779
        #im[:,:,2] -= 123.68
        #im = im/255.0
        #im = im.transpose((2, 0, 1))
        #im = np.expand_dims(im, axis=0)

        # Test pretrained model
        #self.model.compile(optimizer=RMSprop, loss='binary_crossentropy')
        out = self.model.predict(im)
        print(out.shape)
        #print 'predicted label: {}'.format(out[0])
        padding = 4
        for idx, file_name in enumerate(out_list):
            res_mask = np.zeros((37+padding*2, 12+padding*2), np.uint8)

            #heatmap0 = cv2.resize(heatmap0, (self.out_width, self.out_height), interpolation = cv2.INTER_LINEAR)
            for row in range(37):
                for col in range(12):
                    # Empty, misc, Heart, Liver, Lung
                    val_list = [out[idx,0].item(row,col), out[idx,4].item(row,col), out[idx,1].item(row,col), out[idx,2].item(row,col), out[idx,3].item(row,col)]
                    val_list.index(max(val_list))
                    res_mask.itemset((row+padding,col+padding), (0+val_list.index(max(val_list)))*50)
            cv2.imwrite(file_name+"_all.png", cv2.resize(res_mask, (self.out_width, self.out_height), interpolation = cv2.INTER_LINEAR))
            #cv2.imwrite(file_name+"_heart.png", cv2.resize((out[idx,1] * 255).astype('uint8'), (self.out_width, self.out_height), interpolation = cv2.INTER_LINEAR))
            #cv2.imwrite(file_name+"_liver.png", cv2.resize((out[idx,2] * 255).astype('uint8'), (self.out_width, self.out_height), interpolation = cv2.INTER_LINEAR))
            #cv2.imwrite(file_name+"_lung.png", cv2.resize((out[idx,3] * 255).astype('uint8'), (self.out_width, self.out_height), interpolation = cv2.INTER_LINEAR))
            #cv2.imwrite(file_name+"_misc.png", cv2.resize((out[idx,4] * 255).astype('uint8'), (self.out_width, self.out_height), interpolation = cv2.INTER_LINEAR))
            '''
            heatmap0 = out[idx,0] * 255 # Heart
            heatmap1 = out[idx,1] * 255 # Liver
            heatmap2 = out[idx,2] * 255 # Lung
            heatmap3 = out[idx,3] * 255 # Misc
            cv2.imwrite(file_name+"_heart.png", cv2.resize(heatmap0.astype('uint8'), (self.out_width, self.out_height), interpolation = cv2.INTER_LINEAR))
            cv2.imwrite(file_name+"_liver.png", cv2.resize(heatmap1.astype('uint8'), (self.out_width, self.out_height), interpolation = cv2.INTER_LINEAR))
            cv2.imwrite(file_name+"_lung.png", cv2.resize(heatmap2.astype('uint8'), (self.out_width, self.out_height), interpolation = cv2.INTER_LINEAR))
            cv2.imwrite(file_name+"_misc.png", cv2.resize(heatmap3.astype('uint8'), (self.out_width, self.out_height), interpolation = cv2.INTER_LINEAR))
            '''
    '''
    def train(self):
        train_data_dir = '../../../data/medium_vicera_dataset/training'
        validation_data_dir = '../../../data/medium_vicera_dataset/validation'
        nb_train_samples = 3600
        nb_validation_samples = 400
        nb_epoch = 25

        # this is the augmentation configuration we will use for training
        train_datagen = ImageDataGenerator(rescale=1. / 255,
                                           shear_range=0.2,
                                           zoom_range=0.2,
                                           horizontal_flip=True)

        # this is a generator that will read pictures found in
        # subfolers of 'data/train', and indefinitely generate
        # batches of augmented image data
        train_set = train_datagen.flow_from_directory(train_data_dir,
                                                      target_size=(self.img_width, self.img_height),
                                                      batch_size=32,
                                                      class_mode='binary')  # we need binary labels

        # this is the augmentation configuration we will use for testing:
        # only rescaling
        test_datagen = ImageDataGenerator(rescale=1. / 255)

        # this is a similar generator, for validation data
        val_set = test_datagen.flow_from_directory(validation_data_dir,
                                                   target_size=(self.img_width, self.img_height),
                                                   batch_size=32,
                                                   class_mode='binary')  # we need binary labels

        self.model.fit_generator(train_set,
                                 samples_per_epoch=nb_train_samples,
                                 nb_epoch=nb_epoch,
                                 validation_data=val_set,
                                 nb_val_samples=nb_validation_samples)

        self.model.save_weights('weights/small_cnn.h5')  # always save your weights after training or during training

        # serialize model to JSON
        model_json = self.model.to_json()
        with open("model/model.json", "w") as json_file:
            json_file.write(model_json)
    '''
