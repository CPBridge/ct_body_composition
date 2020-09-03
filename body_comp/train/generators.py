import os

import numpy as np

from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from imageio import imread
from scipy.special import expit
from skimage.transform import resize


class SegmentationSequence(Sequence):

    def __init__(self, images, masks, batch_size, jitter=False):
        self.masks = masks
        self.images = images
        self.batch_size = batch_size
        self.shuffled_indices = np.random.permutation(self.images.shape[0])
        self.jitter = jitter
        if self.jitter:
            self.jitter_datagen = ImageDataGenerator(rotation_range=5,
                                                     width_shift_range=0.05,
                                                     height_shift_range=0.05,
                                                     fill_mode="nearest")

    def __len__(self):
        return self.images.shape[0] // self.batch_size

    def __getitem__(self, idx):

        # The shuffled indices in this batch
        batch_inds = self.shuffled_indices[idx * self.batch_size : (idx + 1) * self.batch_size]

        if self.jitter:

            batch_images_list = []
            batch_masks_list = []

            for i in batch_inds:
                # Stack mask and image together to ensure that they are transformed
                # in exactly the same way
                stacked = np.dstack([self.images[i, :, :, :].astype(np.uint8), self.masks[i, :, :, :]])
                transformed = self.jitter_datagen.random_transform(stacked)

                batch_images_list.append(transformed[:, :, 0].astype(float))
                batch_masks_list.append(transformed[:, :, 1])

            batch_images = np.dstack(batch_images_list)
            batch_images = np.transpose(batch_images[:, :, :, np.newaxis], [2, 0, 1, 3])
            batch_masks = np.dstack(batch_masks_list)
            batch_masks = np.transpose(batch_masks[:, :, :, np.newaxis], [2, 0, 1, 3])

        else:

            # Slice images and labels for this batch
            batch_images = self.images[ batch_inds, :, :, :]
            batch_masks = self.masks[ batch_inds, :, :, :]

        return (batch_images, batch_masks)

    def on_epoch_end(self):
        # Shuffle the dataset indices again
        self.shuffled_indices = np.random.permutation(self.images.shape[0])


class SliceSelectionSequence(Sequence):

    def __init__(self, labels, image_dir, batch_size, batches_per_epoch,
                 jitter=False, sigmoid_scale=None):
        self.labels = labels
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.jitter = jitter
        self.sigmoid_scale = sigmoid_scale
        self.shuffled_indices = np.random.permutation(len(labels))
        if self.jitter:
            self.jitter_datagen = ImageDataGenerator(rotation_range=5,
                                                     width_shift_range=0.05,
                                                     height_shift_range=0.05,
                                                     fill_mode="constant",
                                                     cval=0)

    def __len__(self):
        return self.batches_per_epoch

    def __getitem__(self, idx):

        # The shuffled indices in this batch
        batch_inds = self.shuffled_indices[idx * self.batch_size: (idx + 1) * self.batch_size]

        # Labels for this batch
        batch_labels = self.labels[batch_inds]

        # Soft-threshold the distances using a sigmoid
        if self.sigmoid_scale is not None:
            batch_labels = expit(batch_labels / self.sigmoid_scale)

        # The images for this batch
        images_list = []
        for i in batch_inds:

            # Load in image
            filename = os.path.join(self.image_dir, str(i).zfill(6) + '.png')
            im = resize(imread(filename), (256, 256), mode='constant',
                        preserve_range=True, anti_aliasing=True)[:, :, np.newaxis]

            # Apply random jitter (rotation, shift, zoom)
            if self.jitter:
                im = self.jitter_datagen.random_transform(im)

            images_list.append(im)

        batch_images = np.dstack(images_list).astype(float)
        batch_images = np.transpose(batch_images[:, :, :, np.newaxis], [2, 0, 1, 3])

        return (batch_images, batch_labels)

    def on_epoch_end(self):
        # Shuffle the dataset indices again
        required = self.batches_per_epoch * self.batch_size
        use_replacement = required > len(self.labels)
        self.shuffled_indices = np.random.choice(len(self.labels), required, replace=use_replacement)
