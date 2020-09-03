import datetime
import json
import os

import pandas as pd

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.models import model_from_json
import tensorflow as tf

from body_comp.train.generators import SliceSelectionSequence
from body_comp.train.densenet_regression import DenseNet


def train(data_dir, model_output_dir, epochs=100, name=None, batch_size=16,
          gpus=1, learning_rate=0.1, nb_slices=1, threshold=10.0,
          load_weights=None, initial_epoch=0, nb_layers_per_block=4, nb_blocks=4,
          nb_initial_filters=16, growth_rate=12, compression_rate=0.5,
          activation='relu', initializer='glorot_uniform', batch_norm=True):

    args = locals()

    # Set up dataset
    train_image_dir = os.path.join(data_dir, 'images/train')
    val_image_dir = os.path.join(data_dir, 'images/val')
    train_meta_file = os.path.join(data_dir, 'meta/train.csv')
    val_meta_file = os.path.join(data_dir, 'meta/val.csv')
    train_labels = pd.read_csv(train_meta_file)['ZOffset'].values
    val_labels = pd.read_csv(val_meta_file)['ZOffset'].values

    train_generator = SliceSelectionSequence(
        train_labels, train_image_dir, batch_size, 1000, jitter=True, sigmoid_scale=threshold
    )
    val_generator = SliceSelectionSequence(
        val_labels, val_image_dir, batch_size, 50, sigmoid_scale=threshold
    )

    # Directories and files to use
    if name is None:
        name = 'untitled_model_' + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = os.path.join(model_output_dir, name)
    tflow_dir = os.path.join(output_dir, 'tensorboard_log')
    weights_path = os.path.join(output_dir, 'weights-{epoch:02d}-{val_loss:.4f}.hdf5')
    architecture_path = os.path.join(output_dir, 'architecture.json')
    tensorboard = TensorBoard(log_dir=tflow_dir, histogram_freq=0, write_graph=False, write_images=False)

    if load_weights is None:
        os.mkdir(output_dir)
        os.mkdir(tflow_dir)

        args_path = os.path.join(output_dir, 'args.json')
        with open(args_path, 'w') as json_file:
            json.dump(args, json_file, indent=4)

        # Create the model
        print('Compiling model')
        with tf.device('/cpu:0'):
            model = DenseNet(
                img_dim=(256, 256, 1),
                nb_layers_per_block=nb_layers_per_block,
                nb_dense_block=nb_blocks,
                growth_rate=growth_rate,
                nb_initial_filters=nb_initial_filters,
                compression_rate=compression_rate,
                sigmoid_output_activation=True,
                activation_type=activation,
                initializer=initializer,
                output_dimension=nb_slices,
                batch_norm=batch_norm
            )

        # Save the architecture
        with open(architecture_path, 'w') as json_file:
            json_file.write(model.to_json())

    else:
        with open(architecture_path, 'r') as json_file:
            model = model_from_json(json_file.read())

        # Load the weights
        model.load_weights(load_weights)

    # Move to multi GPUs
    # Use multiple devices
    if gpus > 1:
        parallel_model = multi_gpu_model(model, gpus)
        model_checkpoint = MultiGPUModelCheckpoint(weights_path, monitor='val_loss', save_best_only=False)
    else:
        parallel_model = model
        model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=False)

    # Set up the learning rate scheduler
    def lr_func(e):
        print("Learning Rate Update at Epoch", e)
        if e > 0.75 * epochs:
            return 0.01 * learning_rate
        elif e > 0.5 * epochs:
            return 0.1 * learning_rate
        else:
            return learning_rate

    lr_scheduler = LearningRateScheduler(lr_func)

    # Compile multi-gpu model
    loss = 'mean_absolute_error'
    parallel_model.compile(optimizer=Adam(lr=learning_rate), loss=loss)

    print('Starting training...')

    parallel_model.fit_generator(train_generator, epochs=epochs,
                                 shuffle=False, validation_data=val_generator,
                                 callbacks=[model_checkpoint, tensorboard, lr_scheduler],
                                 use_multiprocessing=True,
                                 workers=16,
                                 initial_epoch=initial_epoch)

    return model
