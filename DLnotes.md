# Notes on deep learning #

## Teory ##
Lecture notes for the Stanford class on CNNs: [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io)

http://machinelearningmastery.com/crash-course-recurrent-neural-networks-deep-learning/
http://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
http://machinelearningmastery.com/create-algorithm-test-harness-scratch-python/

## Tools ##
Web service: [Online live supervision of training process](http://aetros.com/)
http://machinelearningmastery.com/machine-learning-performance-improvement-cheat-sheet/

http://machinelearningmastery.com/improve-deep-learning-performance/
http://machinelearningmastery.com/feature-importance-and-feature-selection-with-xgboost-in-python/
http://machinelearningmastery.com/gentle-introduction-xgboost-applied-machine-learning/

http://machinelearningmastery.com/image-augmentation-deep-learning-keras/

http://machinelearningmastery.com/feature-selection-machine-learning-python/

http://machinelearningmastery.com/prepare-data-machine-learning-python-scikit-learn/

http://machinelearningmastery.com/visualize-machine-learning-data-python-pandas/

http://machinelearningmastery.com/understand-machine-learning-data-descriptive-statistics-python/

http://machinelearningmastery.com/bagging-and-random-forest-ensemble-algorithms-for-machine-learning/

http://machinelearningmastery.com/question-to-understand-any-machine-learning-algorithm/
## Keras ##

http://machinelearningmastery.com/using-learning-rate-schedules-deep-learning-models-python-keras/
http://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/
http://machinelearningmastery.com/check-point-deep-learning-models-keras/
Superb guide for getting started with keras and CNNs: [Building powerful image classification models using very little data](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)

Visualizing filers by maximizing output for individual classes(interesting, but not that useful): [Visualizing Deep Neural Networks Classes and Features](http://ankivil.com/visualizing-deep-neural-networks-classes-and-features/)

Guide to setting up a simple NN: [5 Step Life-Cycle for Neural Network Models in Keras](http://machinelearningmastery.com/5-step-life-cycle-neural-network-models-keras/)

Tuning NN hyperparameters using sklearns grid search: [How to Grid Search Hyperparameters for Deep Learning Models in Python With Keras](http://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/)

Storing weights and models: [Save and Load Your Keras Deep Learning Models](http://machinelearningmastery.com/save-load-keras-deep-learning-models/)


**Enumerating layers in the VGG16 model**
```python
model.add(ZeroPadding2D((1,1),input_shape=(3,224,224))) #0
model.add(Convolution2D(64, 3, 3, activation='relu'))   #1
model.add(ZeroPadding2D((1,1)))                         #2
model.add(Convolution2D(64, 3, 3, activation='relu'))   #3
model.add(MaxPooling2D((2,2), strides=(2,2)))           #4

model.add(ZeroPadding2D((1,1)))                         #5
model.add(Convolution2D(128, 3, 3, activation='relu'))  #6
model.add(ZeroPadding2D((1,1)))                         #7
model.add(Convolution2D(128, 3, 3, activation='relu'))  #8
model.add(MaxPooling2D((2,2), strides=(2,2)))           #9

model.add(ZeroPadding2D((1,1)))                         #10
model.add(Convolution2D(256, 3, 3, activation='relu'))  #11
model.add(ZeroPadding2D((1,1)))                         #12
model.add(Convolution2D(256, 3, 3, activation='relu'))  #13
model.add(ZeroPadding2D((1,1)))                         #14
model.add(Convolution2D(256, 3, 3, activation='relu'))  #15
model.add(MaxPooling2D((2,2), strides=(2,2)))           #16

model.add(ZeroPadding2D((1,1)))                         #17
model.add(Convolution2D(512, 3, 3, activation='relu'))  #18
model.add(ZeroPadding2D((1,1)))                         #19
model.add(Convolution2D(512, 3, 3, activation='relu'))  #20
model.add(ZeroPadding2D((1,1)))                         #21
model.add(Convolution2D(512, 3, 3, activation='relu'))  #22
model.add(MaxPooling2D((2,2), strides=(2,2)))           #23

model.add(ZeroPadding2D((1,1)))                         #24
model.add(Convolution2D(512, 3, 3, activation='relu'))  #25
model.add(ZeroPadding2D((1,1)))                         #26
model.add(Convolution2D(512, 3, 3, activation='relu'))  #27
model.add(ZeroPadding2D((1,1)))                         #28
model.add(Convolution2D(512, 3, 3, activation='relu'))  #29
model.add(MaxPooling2D((2,2), strides=(2,2)))           #30
```
## Lasagne and nolearn ##
[Deep learning – Convolutional neural networks and feature extraction with Python](http://blog.christianperone.com/2015/08/convolutional-neural-networks-and-feature-extraction-with-python/)



## TODO ##


## Installing cuda 8.0 Ubuntu ##

https://github.com/saiprashanths/dl-setup

nvidia driver
```bash
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update
sudo apt-get install nvidia-370
```
