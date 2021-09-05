# Running Inference

Inference is the process of running the existing trained model on new data.

### Input Data

The inference process works directly on the data in DICOM format. However, the
DICOM files must be organized into studies in a particular format. The DICOM
files for each study should be placed into a single directory, where the name
of the directory is the MRN of the study followed by the accession number of
the study, separated by an underscore.  For example `0123467_135792468`. Within
the study directory, the names of the DICOM files do not matter.

If you do not have the data placed in this format, there is a script in the
`bin` directory that will created an organized copy of your data for you.

```bash
$ python organize_inference_data.py /path/to/existing_data_directory /path/to/new/organized/directory
```

In addition, there should be a CSV file with columns named `MRN` and `ACC`,
which list the MRN and accession numbers that you wish to process. E.g.,

```
MRN,ACC
012345,678910
54321,109876
314159,26535
```

Alternatively, the CSV file may contain a column named `StudyName`, which
should match the name of the directory in which the files for the study are
stored if it uses some convention other than `MRN_ACC`.

### Running the Process

The `run_from_csv.py` script is the main script for performing inference. It
takes in a CSV file of MRNs and ACCs or *StudyNames* (described above), and
data directories, runs inference on every study, including the series
selection, slice selection, and segmentation steps, and then outputs the
results into a results directory.

Basic usage looks like this:

```bash
$ python run_from_csv.py my_csv_file.csv config.json /path/to/results/directory /path/to/input/directory1 /path/to/input/directory2
```

Note that there can be an arbitrary number (one or more) of input directories,
each with the studies laid out in the `MRN_ACC` format. This allows for
processing data from multiple data pulls at once.

See notes below for more information on the config file to set up the
configuration file before running.

At the end of the process, the results directory will contain several artifacts:

`json_files/` - This directory contains the full results from the body
composition analysis in JSON format.  There is one file per study, named with
the study `{MRN}_{ACC}.json` (or `{StudyName}.json` if that convention was used
in the input csv file).

`previews/` - This directory contains preview images that may be used for
efficient visual checking of the results. There is one preview png file per
*series* that was successfully processed by the algorithm (there are often
multiple such series per study). Each file is named
`{MRN}_{ACC}_{SeriesInstanceUID}_preview.png` (or
`{StudyName}_{SeriesInstanceUID}_preview.png`).

`run_log.csv` - This file contains the basic results from running the
model.  For each study listed in the input csv file, it lists whether the
relevant DICOM data was found successfully, how many series from this study (if
any) the model was able to run on successfully.

`summary.csv` - This file lists a summary of the results of the results on
every *series* successfully processed by the model, including the area of the
body composition compartments and their Hounsfield unit statistics. It is a
subset of the information in the JSON files.

`filtered_summary.csv` - This file consists of a subset of the rows of the
`summary.csv` file after a filtering process has been applied. The filtering
process removes likely slice selection failures, and selects the most
appropriate series per study if there are multiple series that successfully ran
for a particular study.


### Config File

The process requires a configuration file in the JSON format to set up certain
parameters of the inference process. The default configuration file is provided
in the configs subdirectory of the package. You will need to edit this to
specify the locations of your trained models.

Here is the layout of an example configuration file:

```json
{
    "sigmoid_output": true,
    "slice_selection_weights": "/path/to/some/model.hdf5",
    "slice_params": {
        "L3": {
            "slice_selection_model_output_index": 0,
            "class_names": [
                "muscle",
                "subcutaneous_fat",
                "visceral_fat"
                ],
            "model_weights": "/path/to/some/other_model.hdf5",
            "regression_plot_colour": "red"
        }
    }
}
```

Explanations of each of these parameters are found below:

```
slice_selection_weights: string
    Path to hdf file containing the model weights for the slice selection model.
slice_params: dict
    Python dictionary containing a mapping from string (slice name) to dictionary.
    The lower dictionary has the following entries:
        'slice_selection_model_output_index': int
            Which output of the slice selection model relates to this slice (only relevant if sigmoid_output is
            True)
        'z_pos': float
            The position of this slice in the latent space (only relevant if sigmoid_output is False)
        'class_names': list of strings
            Names of the classes present in the segmentation model of this slice, excluding the background
            class. Should match the number of channels of the segmentation model (-1 due to background class).
            List should be in increasing order of channel index.
        'model_weights': string
            Path to hdf file containing the model weights for the segmentation model to use for this slice.
        'regression_plot_colour': string
            A colour (as recognised by a matplotlib plot command) to use for this slice on the output regression
            plots
    The default value of ``slice_params`` is configured to run on an L3 slice to segment three compartments:
    muscle, visceral fat, and subcutaneous fat.
sigmoid_output: bool
    Set to true if the slice selection model outputs a true sigmoid range (between 0 and 1 rather than -1
    and 1) for each target slice. If false, the slice selection model outputs a single number in a 1D space for
    each input slice, which is compared to the 'z_pos' field of 'slice_params' to perform slice selection.
```


### Additional Options

There are a number of further options to customize the behavior of this process.
They may be passed as command-line arguments to the `run_from_csv.py` file:

`--num_threads`, `-t` - The number of parallel threads to use to read in DICOM
files (typically the most time intensive step of the processing especially if the
files are being read from a remote file system). You should choose this appropriately
based on your hardware. Using a higher number if multiple CPU cores are available
will usually speed up processing significantly.

`--segmentation_range`, `-r` - The range either side of the selected slice (in
mm) to perform segmentation on for multislice analysis. If this is specified,
then the slices are selected as usual, but then any slice that lies within the
given distance of the selected slice is segmented, and the results are
averaged. This usually gives a more robust result, but setting it too high will
cause the model to segment areas that it wasn't trained to segment. If this
option is chosen, there is an additional sub-directory of the output directory
called `all_slices`, which stores segmentation masks and original images for
every output slice in `.png` format in sub-directories named by
`{MRN}_{ACC}_{SeriesInstanceUID}`. The filename of the `.png` images matches
that the position in the JSON file. The preview image contains just the chosen
center slice.

`--use_directory_list`, `-l` - If there are a large number of input directories
that need to be searched to find the studies listed in the CSV file, it may
be more convenient to list the input directories in a plain text file. To do this
 create a plain text file with one input directory per line, and specify the path
 to that file in place of the input directory when running the process. Then use
 this flag to enable the behavior.

`--keep_existing`, `-k` - This flag is used to carry on a process that was
previously interrupted. If the same output directory as a previous run is used,
the process will not re-process studies that already have results.

`--rerun_exceptions`, `-e` - This flag is only used in combination with
`--keep_existing`. If specified, all studies that were previously processed but
failed with an exception will be run again (in addition to any studies that were
not previously processed).

`--recursive`, `-R` - In this case, all sub-directories of the study directories
will be searched for DICOM files belonging to that study.

`--dicom_seg`, `-d` - If specified, the segmentations will also be output in
DICOM segmentation format. This is a standard format that can be read and
displayed by some DICOM viewers.

`--min_slices_per_series`, `-m` - Reject series with fewer than this number of
images. Useful for rejecting small localizer series. Default: 20
