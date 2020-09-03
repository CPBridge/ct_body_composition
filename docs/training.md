# Model Training

The two neural networks that comprise the body composition estimator (the
segmentation model and the slice selection model) are trained independently
using two different scripts.

## Segmentation Model

#### Preparing Training Data

Before training the segmentation model you must prepare the training data in a
pre-specified format. Training data consists of a set of 2D CT slices and
corresponding segmentation masks. Each should be prepared as a numpy array
stored in a `.npy` file as follows, and all files should be placed in a single
directory.

`train_images.npy` -- A numpy array of size (*N* x 512 x 512 x 1), where *N* is
the number of training samples. This represents all *N* training CT images
stacked down the first dimension of the array, and a singleton channel
dimension at the end. All images must be resized to 512 x 512 if they are
originally a different size. The pixel intensities should be raw Hounsfield
units, without intensity windowing or scaling. The data type can be any data
type that is capable of representing the full range of CT values, i.e. *signed*
integers with 16 bits or greater, or floating point numbers. These images are
used to train the model.

`val_images.npy` -- An array of validation images that are used to monitor the
progress of the training process and compare the generalization performance of
different models. Its construction is identical to `train_images.npy` (note
that the number of images in the validation will usually different to the
number of images in the training set).

`train_masks.py` -- An array the same shape as `train_images.npy`, where all
spatial dimensions correspond to the `train_images.npy` array. Each slice of
the masks array is the segmentation mask for the same slice in the images
array. The masks should have a `uint8` data type, and each pixel encodes the
segmentation label of the corresponding pixel in the image array. A value of 0
denotes the background class, 1 denotes the 'muscle' class, 2 denotes the
'subcutaneous fat' class, and 3 denotes the 'visceral fat' class.

`val_masks.py` -- Mask array for the validation images in `val_images.npy`.
Construction is otherwise identical to `train_masks.npy`.

#### Training the Model

The segmentation model is trained with the `train_segmentation.py` script in
the `bin` directory. You can run the basic training routine by passing the two
required arguments:

`data_dir` -- Directory in which the training data (`train_images.npy`,
`val_images.npy`, `train_masks.npy` and `train_masks.npy`) arrays are stored.

`model_output_dir` -- The model checkpoints and associated files will be stored
in a sub-directory of this directory.

For example:

```bash
$ python train_segmentation.py /path/to/data/ /path/to/models/
```

There are a number of other options you can specify to tweak the model
architecture and training procedure. Of particular note are:

* `-a` - Specify the name of the model (name of the output directory)
* `-g` - Specify the number of GPUs to use for training
* `-l` - Specify the initial learning rate
* `-b` - Specify the batch size

Run the help for a full list of options:

```bash
$ python3 train_segmentation.py --help
```


## Slice Selection Model

#### Preparing Training Data

The training data for the slice selection model consists of CT slices from the
entire chest/abdomen/pelvis region, with a physical offset from a known L3
slice. To allow for efficient loading during training, the images should be
extracted out into .png format.

Each .png image should have pixel values between 0 and 255 as a result of
intensity clipping the raw Hounsfield units with a window level of 40 and a
window width of 400 and rescaling that range to 0 to 255. I.e. raw pixel values
below -160HU are transformed to a pixel value of 0, raw pixel values above
240HU are transformed to 255, and raw pixel vales between -160HU and 240HU
should be transformed into a value between 0 and 255 (inclusive) with linear
scaling.

The extracted slices should be divided into train, test and validation splits
and placed in a directory according to the split. Within each directory, the
slices should be named according to a six-digit integer ID starting at 000000
and covering all integers between 0 and *N-1*, where *N* is the number of slices
within that split.

Each split should be accompanied by a CSV file that contains the offset from
the L3 slice in a column called `ZOffset`. The *i*th row of the CSV (starting
at index 0 after the header row) should contain the *z*-offset for the slice
with ID *i*. The *z*-offset for a slice should represent its offset above or
below the L3 slice in mm in the physical space of the scanner. Slices above the
chosen L3 slice (closer to the head) should be given positive offsets, and
slices below the chosen L3 slice (closer to the feet) should be given negative
offsets.  Other columns may be included in this CSV file (e.g. information to
map back to the original series). Such columns will be ignored by the training
process. An example CSV file could look like this:

```
,SOPInstanceUID,ZOffset
0,1.2345.678,-234.6
1,1.2345.789,5.2
2,1.2345.987,145.3
```

The files described above should be placed within a directory with the
following structure:


```
|- images/
|  |- train/
|  |  |- 000000.png
|  |  |- 000001.png
|  |  |- 000002.png
|  |  |- ...
|  |  |- 010000.png
|  |
|  |- val/
|     |- 000000.png
|     |- 000001.png
|     |- 000002.png
|     |- ...
|     |- 001000.png
|
|- meta/
   |- train.csv
   |- val.csv

```


#### Training the Model

The script `train_slice_selection.py` in the `bin` directory is used to train
the slice selection model. The required arguments are the location of the
training data directory described above and the location to place the trained
model.

```bash
$ python train_slice_selection.py /path/to/data/ /path/to/models/
```

A number of optional arguments may be passed to control various aspects of the
model architecture and training process. Of particular note are:

* `-a` - Specify the name of the model (name of the output directory)
* `-g` - Specify the number of GPUs to use for training
* `-l` - Specify the initial learning rate
* `-b` - Specify the batch size


Run the help for a full list of options:

```bash
$ python3 train_slice_selection.py --help
```
