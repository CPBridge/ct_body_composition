import datetime
import json
import os

import numpy as np

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.callbacks import ModelCheckpoint

from body_comp.train.unet import get_unet_2D
from body_comp.train.losses import dice_coef_multiclass_loss_2D
from body_comp.train.callbacks import MultiGPUModelCheckpoint
from body_comp.train.generators import SegmentationSequence


# Windowing function
def apply_window(image, win_centre, win_width):
    range_bottom = win_centre - win_width / 2
    scale = 256 / win_width
    image = image - range_bottom

    image = image * scale

    image[image < 0] = 0
    image[image > 255] = 255

    return image


def train(data_dir, model_output_dir, name=None, epochs=100, batch_size=16, load_weights=None,
          gpus=1, learning_rate=0.1, decay_half_time=20.0, apply_window_function=False, num_convs=2,
          activation='relu', compression_channels=[32, 64, 128, 256, 512], decompression_channels=[256, 128, 64, 32]):

    args = locals()

    train_images_file = os.path.join(data_dir, 'train_images.npy')
    val_images_file = os.path.join(data_dir, 'val_images.npy')
    train_masks_file = os.path.join(data_dir, 'train_masks.npy')
    val_masks_file = os.path.join(data_dir, 'val_masks.npy')

    images_train = np.load(train_images_file)
    images_train = images_train.astype(float)
    images_val = np.load(val_images_file)
    images_val = images_val.astype(float)
    masks_train = np.load(train_masks_file)
    masks_train = masks_train.astype(np.uint8)
    masks_val = np.load(val_masks_file)
    masks_val = masks_val.astype(np.uint8)

    if apply_window_function:
        images_train = apply_window(images_train, 40, 400)
        images_val = apply_window(images_val, 40, 400)

    # Directories and files to use
    if name is None:
        name = 'untitled_model_' + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = os.path.join(model_output_dir, name)
    tflow_dir = os.path.join(output_dir, 'tensorboard_log')
    os.mkdir(output_dir)
    os.mkdir(tflow_dir)
    weights_path = os.path.join(output_dir, 'weights-{epoch:02d}-{val_loss:.4f}.hdf5')
    architecture_path = os.path.join(output_dir, 'architecture.json')
    tensorboard = TensorBoard(log_dir=tflow_dir, histogram_freq=0, write_graph=False, write_images=False)

    args_path = os.path.join(output_dir, 'args.json')
    with open(args_path, 'w') as json_file:
        json.dump(args, json_file, indent=4)


    print('Creating and compiling model...')
    model = get_unet_2D(
        4,
        (512, 512, 1),
        num_convs=num_convs,
        activation=activation,
        compression_channels=compression_channels,
        decompression_channels=decompression_channels
    )

    # Save the architecture
    with open(architecture_path,'w') as json_file:
        json_file.write(model.to_json())

    # Use multiple devices
    if gpus > 1:
        parallel_model = multi_gpu_model(model, gpus)
    else:
        parallel_model = model

    # Should we load existing weights?
    if load_weights is not None:
        print('Loading pre-trained weights...')
        parallel_model.load_weights(load_weights)

    val_batches = images_val.shape[0] // batch_size
    train_batches = images_train.shape[0] // batch_size

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

    train_generator = SegmentationSequence(images_train, masks_train, batch_size, jitter=True)
    val_generator = SegmentationSequence(images_val, masks_val, batch_size)

    parallel_model.compile(optimizer=Adam(lr=learning_rate), loss=dice_coef_multiclass_loss_2D)

    print('Fitting model...')
    if gpus > 1:
        model_checkpoint = MultiGPUModelCheckpoint(weights_path, monitor='val_loss', save_best_only=False)
    else:
        model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=False)
    parallel_model.fit_generator(train_generator, train_batches, epochs=epochs,
              shuffle=False, validation_steps=val_batches, validation_data=val_generator, use_multiprocessing=True,
              workers=10, max_queue_size=40, callbacks=[model_checkpoint, tensorboard, lr_scheduler])

    # Save the template model weights
    model.save_weights(os.path.join(output_dir, 'final_model.hdf5'))

    return model
