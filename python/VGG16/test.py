
def get_model(trainer):
    """Returns the model (not compiled, not trained)

    Parameters
    ----------
    trainer : aetros.Trainer.Trainer
    """

    import keras
    import keras.models
    import keras.layers
    import keras.layers.convolutional
    import keras.layers.core

    input = keras.layers.Input(shape=(3, 224, 224), name="Input")

    # Group: Conv Block 1
    zero_padding_1 = keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), name="zero_padding_1")(input)
    conv1_1 = keras.layers.convolutional.Convolution2D(64, 3, 3, border_mode='valid', name="conv1_1", activation='relu', init='glorot_uniform')(zero_padding_1)
    zero_padding_2 = keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), name="zero_padding_2")(conv1_1)
    conv1_2 = keras.layers.convolutional.Convolution2D(64, 3, 3, border_mode='valid', name="conv1_2", activation='relu', init='glorot_uniform')(zero_padding_2)
    pool_1 = keras.layers.convolutional.MaxPooling2D(pool_size=(2, 2), name="pool_1")(conv1_2)

    # Group: Conv Block 2
    zero_padding_3 = keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), name="zero_padding_3")(pool_1)
    conv2_1 = keras.layers.convolutional.Convolution2D(128, 3, 3, border_mode='valid', name="conv2_1", activation='relu', init='glorot_uniform')(zero_padding_3)
    zero_padding_4 = keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), name="zero_padding_4")(conv2_1)
    conv2_2 = keras.layers.convolutional.Convolution2D(128, 3, 3, border_mode='valid', name="conv2_2", activation='relu', init='glorot_uniform')(zero_padding_4)
    pool_2 = keras.layers.convolutional.MaxPooling2D(pool_size=(2, 2), name="pool_2")(conv2_2)

    # Group: Conv Block 3
    zero_padding_5 = keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), name="zero_padding_5")(pool_2)
    conv3_1 = keras.layers.convolutional.Convolution2D(256, 3, 3, border_mode='valid', name="conv3_1", activation='relu', init='glorot_uniform')(zero_padding_5)
    zero_padding_6 = keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), name="zero_padding_6")(conv3_1)
    conv3_2 = keras.layers.convolutional.Convolution2D(256, 3, 3, border_mode='valid', name="conv3_2", activation='relu', init='glorot_uniform')(zero_padding_6)
    zero_padding_7 = keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), name="zero_padding_7")(conv3_2)
    conv3_3 = keras.layers.convolutional.Convolution2D(256, 3, 3, border_mode='valid', name="conv3_3", activation='relu', init='glorot_uniform')(zero_padding_7)
    pool_3 = keras.layers.convolutional.MaxPooling2D(pool_size=(2, 2), name="pool_3")(conv3_3)

    # Group: Conv Block 4
    zero_padding_8 = keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), name="zero_padding_8")(pool_3)
    conv4_1 = keras.layers.convolutional.Convolution2D(512, 3, 3, border_mode='valid', name="conv4_1", activation='relu', init='glorot_uniform')(zero_padding_8)
    zero_padding_9 = keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), name="zero_padding_9")(conv4_1)
    conv4_2 = keras.layers.convolutional.Convolution2D(512, 3, 3, border_mode='valid', name="conv4_2", activation='relu', init='glorot_uniform')(zero_padding_9)
    zero_padding_10 = keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), name="zero_padding_10")(conv4_2)
    conv4_3 = keras.layers.convolutional.Convolution2D(512, 3, 3, border_mode='valid', name="conv4_3", activation='relu', init='glorot_uniform')(zero_padding_10)
    pool_4 = keras.layers.convolutional.MaxPooling2D(pool_size=(2, 2), name="pool_4")(conv4_3)

    # Group: Conv Block 5
    zero_padding_11 = keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), name="zero_padding_11")(pool_4)
    conv5_1 = keras.layers.convolutional.Convolution2D(512, 3, 3, border_mode='valid', name="conv5_1", activation='relu', init='glorot_uniform')(zero_padding_11)
    zero_padding_12 = keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), name="zero_padding_12")(conv5_1)
    conv5_2 = keras.layers.convolutional.Convolution2D(512, 3, 3, border_mode='valid', name="conv5_2", activation='relu', init='glorot_uniform')(zero_padding_12)
    zero_padding_13 = keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), name="zero_padding_13")(conv5_2)
    conv5_3 = keras.layers.convolutional.Convolution2D(512, 3, 3, border_mode='valid', name="conv5_3", activation='relu', init='glorot_uniform')(zero_padding_13)
    pool_5 = keras.layers.convolutional.MaxPooling2D(pool_size=(2, 2), name="pool_5")(conv5_3)
    flatten_1 = keras.layers.core.Flatten(name="flatten_1")(pool_5)

    # Group: Dense Block
    dense_1 = keras.layers.core.Dense(4096, name="dense_1", activation='relu', init='glorot_uniform')(flatten_1)
    dense_1 = keras.layers.core.Dropout(0.5, name="dense_1_dropout")(dense_1)
    dense_2 = keras.layers.core.Dense(4096, name="dense_2", activation='relu', init='glorot_uniform')(dense_1)
    dense_2 = keras.layers.core.Dropout(0.5, name="dense_2_dropout")(dense_2)
    dense_3 = keras.layers.core.Dense(trainer.output_size, name="dense_3", activation='softmax', init='glorot_uniform')(dense_2)

    return keras.models.Model([input], [dense_3])



def compile(trainer, model, loss, optimizer):
    """Compiles the given model (from get_model) with given loss (from get_loss) and optimizer (from get_optimizer)

    Parameters
    ----------
    trainer : aetros.Trainer.Trainer
    """

    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])



def get_training_data(trainer, dataset):
    """Returns the training and validation data from dataset. Since dataset is living in its own domain
    it could be necessary to transform the given dataset to match the model's input layer. (e.g. convolution vs dense)

    Parameters
    ----------
    trainer : aetros.Trainer.Trainer

    dataset: dict
        Contains for each used dataset a key => dictionary, where key is the dataset id and the dict is returned from the Dataset "get_data" method.
        Their keys: 'X_train', 'Y_train', 'X_test', 'Y_test'.
        Example data['mnistExample']['X_train'] for a input layer that has 'mnistExample' as chosen dataset.
    """

    return {
        'x': {'Input': dataset['markpp/dataset/dog_cat_small']['X_train']},
        'y': {'dense_3': dataset['markpp/dataset/dog_cat_small']['Y_train']}
    }


def get_validation_data(trainer, dataset):
    """Returns the training and validation data from dataset.

    Parameters
    ----------
    trainer : aetros.Trainer.Trainer

    dataset: dict
        Contains for each used dataset a key => dictionary, where key is the dataset id and the dict is returned from the Dataset "get_data" method.
        Their keys: 'X_train', 'Y_train', 'X_test', 'Y_test'.
        Example data['mnistExample']['X_test'] for a input layer that has 'mnistExample' as chosen dataset.
    """

    return {
        'x': {'Input': dataset['markpp/dataset/dog_cat_small']['X_test']},
        'y': {'dense_3': dataset['markpp/dataset/dog_cat_small']['Y_test']}
    }

def get_optimizer(trainer):
    """Returns the optimizer

    Parameters
    ----------
    trainer : aetros.Trainer.Trainer
    """

    import keras.optimizers

    optimizer = keras.optimizers.SGD(lr=0.01, decay=0.0001, momentum=0.001, nesterov=False)
    
    return optimizer

def get_loss(trainer):
    """Returns the optimizer

    Parameters
    ----------
    trainer : aetros.Trainer.Trainer
    """

    loss = {'dense_3': 'categorical_crossentropy'}
    return loss


def train(trainer, model, training_data, validation_data):

    """Returns the model (not build, not trained)

    Parameters
    ----------
    trainer : aetros.Trainer.Trainer
    """
    nb_epoch = trainer.settings['epochs']
    batch_size = trainer.settings['batchSize']

    if trainer.has_generator(training_data['x']):
        model.fit_generator(
            trainer.get_first_generator(training_data['x']),
            samples_per_epoch=trainer.samples_per_epoch, 
            nb_val_samples=trainer.nb_val_samples,
            
            nb_epoch=nb_epoch,
            verbose=0,
            validation_data=trainer.get_first_generator(validation_data['x']), 
            callbacks=trainer.callbacks
        )
    else:
        model.fit(
            training_data['x'],
            training_data['y'],
            batch_size=batch_size,
            nb_epoch=nb_epoch,
            verbose=0,
            validation_data=(validation_data['x'], validation_data['y']),
            callbacks=trainer.callbacks
        )
