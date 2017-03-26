import numpy as np
from keras.layers.core import  Lambda, Merge
from keras.layers.convolutional import Convolution2D
from keras import backend as K
from keras.engine import Layer
from os import listdir
from os.path import isfile, join, dirname
from scipy.io import loadmat
import gc
from keras.utils.layer_utils import layer_from_config
from keras.models import Model
from keras.layers import *

# Credits to heuritech for their great code which was a great inspiration.
# Some of the code comes directly from their repository.
# You can look it up: https://github.com/heuritech/convnets-keras



meta_clsloc_file = join(dirname(__file__), "data", "meta_clsloc.mat")


synsets = loadmat(meta_clsloc_file)["synsets"][0]

synsets_imagenet_sorted = sorted([(int(s[0]), str(s[1][0])) for s in synsets[:1000]],
                                 key=lambda v:v[1])

corr = {}
for j in range(1000):
    corr[synsets_imagenet_sorted[j][0]] = j

corr_inv = {}
for j in range(1,1001):
    corr_inv[corr[j]] = j
	
def depthfirstsearch(id_, out=None):
    if out is None:
        out = []
    if isinstance(id_, int):
        pass
    else:
        id_ = next(int(s[0]) for s in synsets if s[1][0] == id_)
        
    out.append(id_)
    children = synsets[id_-1][5][0]
    for c in children:
        depthfirstsearch(int(c), out)
    return out
	
# This is to find all the outputs that correspond to the class we want.
def synset_to_dfs_ids(synset):
    ids = [x for x in depthfirstsearch(synset) if x <= 1000]
    ids = [corr[x] for x in ids]
    return ids
	
# Keras doesn't have a 4D softmax. So we need this.
class Softmax4D(Layer):
    def __init__(self, axis=-1,**kwargs):
        self.axis=axis
        super(Softmax4D, self).__init__(**kwargs)

    def build(self,input_shape):
        pass

    def call(self, x,mask=None):
        e = K.exp(x - K.max(x, axis=self.axis, keepdims=True))
        s = K.sum(e, axis=self.axis, keepdims=True)
        return e / s

    def get_output_shape_for(self, input_shape):
        return input_shape
		

def get_dim(model, layer_index, input_shape=None):
    
    # Input shape is the shape of images used during training.
    if input_shape is not None:
        dummy_vector = np.zeros((1,) + input_shape)
    else:
        if model.layers[0].input_shape[2] is None:
            raise ValueError('You must provide \"input_shape = (3,256,256)\" for example when calling the function.')
        dummy_vector = np.zeros((1,) + model.layers[0].input_shape[1:])
    
    intermediate_layer_model = Model(input=model.input,
                                 output=model.layers[layer_index].output)
    
    out = intermediate_layer_model.predict(dummy_vector)
    
    return out.shape[1:]
	

def from_config(layer, config_dic):
    config_correct = {}
    config_correct['class_name'] = type(layer)
    config_correct['config'] = config_dic
    return layer_from_config(config_correct)
	

def add_to_model(x, layer):
    new_layer = from_config(layer, layer.get_config())
    x = new_layer(x)
    if layer.get_weights() is not None:
        new_layer.set_weights(layer.get_weights())
    return x
	

def layer_type(layer):
    return str(layer)[10:].split(" ")[0].split(".")[-1]
	

def detect_configuration(model):
    # must return the configuration and the number of the first pooling layer
    
    # Names (types) of layers from end to beggining
    inverted_list_layers = [layer_type(layer) for layer in model.layers[::-1]]
    
    layer1 = None
    layer2 = None 
    
    i = len(model.layers)
    
    for layer in inverted_list_layers:
        i -= 1
        if layer2 is None:
            if layer == "GlobalAveragePooling2D" or layer == "GlobalMaxPooling2D":
                layer2 = layer

            elif layer == "Flatten":
                return "local pooling - flatten", i-1
            
        else:
            layer1 = layer
            break
            
    if layer1 == "MaxPooling2D" and layer2 == "GlobalMaxPooling2D":
        return "local pooling - global pooling (same type)", i
    elif layer1 == "AveragePooling2D" and layer2 == "GlobalAveragePooling2D":
        return "local pooling - global pooling (same type)", i
    
    elif layer1 == "MaxPooling2D" and layer2 == "GlobalAveragePooling2D":
        return "local pooling - global pooling (different type)", i+1
    elif layer1 == "AveragePooling2D" and layer2 == "GlobalMaxPooling2D":
        return "local pooling - global pooling (different type)", i+1
    
    else:
        return "global pooling", i
		
    
def insert_weights(layer, new_layer):
    W,b = layer.get_weights()
    n_filter,previous_filter,ax1,ax2 = new_layer.get_weights()[0].shape
    ax1 = ax2 = int(np.sqrt(layer.get_weights()[0].shape[0]/new_layer.get_weights()[0].shape[1]))
    new_W = W.reshape((previous_filter,ax1,ax2,n_filter))
    new_W = new_W.transpose((3,0,1,2))
    new_W = new_W[:,:,::-1,::-1]
    
    new_layer.set_weights([new_W,b])
	
    
def copy_last_layers(model, begin,x):
    
    i=begin
    
    for layer in model.layers[begin:]:
        if layer_type(layer) == "Dense":
            
            if i == len(model.layers)-1:
                x = add_reshaped_layer(layer,x,1, no_activation=True)
            else:
                x = add_reshaped_layer(layer,x,1)
            
        elif layer_type(layer) == "Dropout":
            pass
                
        elif layer_type(layer) == "Activation" and i == len(model.layers)-1:
            break
               
        else:
            x = add_to_model(x, layer)
        i+=1
    
    x = Softmax4D(axis=1,name="softmax")(x)
    return x
    
                
def add_reshaped_layer(layer, x, size, no_activation=False, atrous_rate = None):

    conf = layer.get_config()
    
    if no_activation:
        activation="linear"
    else:
        activation=conf["activation"]
    
    if size == 1:
        new_layer = Convolution2D(conf["output_dim"],size,size, 
                                  activation=activation, name=conf['name'])
    else:
        new_layer = AtrousConvolution2D(conf["output_dim"], size, size, 
                            atrous_rate=(atrous_rate, atrous_rate), 
                            activation=activation, border_mode='valid')
        
    x= new_layer(x)
    # We transfer the weights:
    insert_weights(layer, new_layer)
    return x
    

def to_heatmap(model, input_shape = None):
    
    # there are four configurations possible:
    # global pooling
    # local pooling - flatten
    # local pooling - global pooling (same type)
    # local pooling - global pooling (different type)
    
    model_type, index = detect_configuration(model)
    
    print("Model type detected: " + model_type)
    
    #new_layer.set_weights(model.layers[0].get_weights())
    img_input = Input(shape=(3,None,None))
   
    # Inchanged part:
    middle_model = Model(input=model.layers[1].input, output=model.layers[index-1].output)
    
    x = middle_model(img_input)
    
    print("Model cut at layer: " + str(index))
        
    if model_type == "global pooling":
        x = copy_last_layers(model, index+1,x)
              
    elif model_type == "local pooling - flatten":
        
        layer = model.layers[index]
        dic = layer.get_config()
        atrous_rate = dic["strides"][0] 
        dic["strides"] = (1,1)
        new_pool = from_config(layer, dic)
        x = new_pool(x)
        
        size = get_dim(model, index, input_shape)[1]
        print("Pool size infered: " + str(size))
        
        
        if index+2 != len(model.layers)-1:
            x = add_reshaped_layer(model.layers[index+2], x, size, atrous_rate=atrous_rate)
        else:
            x = add_reshaped_layer(model.layers[index+2], x, size, atrous_rate=atrous_rate, no_activation=True)
            
        x = copy_last_layers(model, index+3,x)
        
        
    elif model_type == "local pooling - global pooling (same type)":
        
        
        dim = get_dim(model, index, input_shape=input_shape)

        new_pool_size = model.layers[index].get_config()["pool_size"][0] * dim[1]
        
        print("Pool size infered: " + str(new_pool_size))
        
        x = AveragePooling2D(pool_size=(new_pool_size, new_pool_size), strides=(1,1)) (x)
        x = copy_last_layers(model, index+2,x)
        
        
    elif model_type == "local pooling - global pooling (different type)":
        x= copy_last_layers(model, index+1,x)
    else:
        raise IndexError("no type for model: " + str(model_type))
        
    
    return Model(img_input, x)
    