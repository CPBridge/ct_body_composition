import functools
import multiprocessing as mp
import os
from collections import defaultdict

from tensorflow.keras.models import load_model

from highdicom.content import AlgorithmIdentificationSequence
from highdicom.seg.content import SegmentDescription
from highdicom.seg.enum import (
    SegmentAlgorithmTypeValues,
    SegmentationTypeValues
)
from highdicom.seg.sop import Segmentation
from highdicom.sr.coding import CodedConcept

import numpy as np

from pkg_resources import resource_filename

import pydicom
from pydicom.sr.codedict import codes
from pydicom.uid import ExplicitVRLittleEndian

from scipy.ndimage.filters import gaussian_filter
from scipy.special import expit

from skimage.transform import resize

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


MODEL_NAME = 'CCDS Body Composition Estimation'
MANUFACTURER = 'MGH & BWH Center for Clinical Data Science'
SERIAL_NUMBER = '1'


KNOWN_SEGMENT_DESCRIPTIONS = {
    'muscle': {
        'segment_label': 'Muscle',
        'segmented_property_category': codes.SCT.Muscle,
        'segmented_property_type': codes.SCT.SkeletalMuscle,
    },
    'subcutaneous_fat': {
        'segment_label': 'Subcutaneous Fat',
        'segmented_property_category': codes.SCT.BodyFat,
        'segmented_property_type': CodedConcept('727176007', 'SCT', 'Entire subcutaneous fatty tissue'),
    },
    'visceral_fat': {
        'segment_label': 'Visceral Fat',
        'segmented_property_category': codes.SCT.BodyFat,
        'segmented_property_type': CodedConcept('725274000', 'SCT', 'Entire adipose tissue of abdomen'),
    }
}


DEFAULT_SLICE_PARAMS = {
    "L3": {
        "slice_selection_model_output_index": 0,
        "class_names": [
            "muscle",
            "subcutaneous_fat",
            "visceral_fat"
        ],
        "model_weights": None,
        "regression_plot_colour": "red"
    }
}


# An exception class to raise when there are (potential) issues with DICOM decompression
class DICOMDecompressionError(Exception):
    pass


class BodyCompositionEstimator:
    """A class that encapsulates the models and processes required to perform body composition analysis on
    CT images.
    """

    def __init__(self,
                 slice_selection_weights=None,
                 slice_params=DEFAULT_SLICE_PARAMS,
                 sigmoid_output=True,
                 num_threads=15,
                 min_slices_per_series=20,
                 algorithm_version='0.1.0'):
        """ Initiate object with the models required

        Parameters:
        -----------
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
        min_slices_per_series: int
            Reject any series with fewer than this number of slices (useful for removing localizers).
        algorithm_version: str
            The algorithm version string used in the DICOM segmentation output.
        num_threads: int
            Number of threads to use (using python multiprocessing) to read in image files.

        """
        self.slice_smoothing_kernel = 2.0
        self.choose_superior = True
        if slice_selection_weights is None:
            raise ValueError("Configuration does not specify a trained slice selection model")
        self.slice_selection_model = load_model(slice_selection_weights, compile=False)

        # Read in all the segmentation models required
        self.slice_params = slice_params.copy()
        for s in self.slice_params.keys():
            if self.slice_params[s]['model_weights'] is None:
                raise ValueError(f"Configuration does not specify a trained segmentation model for class {s}")
        unique_seg_models = list(set([v['model_weights'] for v in self.slice_params.values()]))
        segmentation_models = [load_model(weights, compile=False) for weights in unique_seg_models]
        for s in self.slice_params.keys():
            model_index = unique_seg_models.index((self.slice_params[s]['model_weights']))
            self.slice_params[s]['model'] = segmentation_models[model_index]
            self.slice_params[s]['segmentation_input_shape'] = segmentation_models[model_index].input_shape[1:3]

        self.slice_selection_input_shape = self.slice_selection_model.input_shape[1:3]
        self.win_centre = 40.0
        self.win_width = 400
        self.sigmoid_scale = 10.0
        self.sigmoid_output = sigmoid_output
        self.min_slices_per_series = min_slices_per_series
        self.num_threads = num_threads
        self.algorithm_version = algorithm_version

        # Dictionaries of tags at instances, series and study to put into the results
        self.instance_level_tags = {
            'tube_current_mA': {'keyword': 'XRayTubeCurrent', 'type': float},
            'exposure_mAs': {'keyword': 'Exposure', 'type': float},
            'exposure_time_ms': {'keyword': 'ExposureTime', 'type': float},
            'kvp': {'keyword': 'KVP', 'type': float},
        }

        self.series_level_tags = {
            'slice_thickness_mm': {'keyword': 'SliceThickness', 'type': float},
            'reconstruction_kernel': {'keyword': 'ConvolutionKernel', 'type': str},
            'contrast_bolus_agent': {'keyword': 'ContrastBolusAgent', 'type': str},
            'contrast_bolus_ingredient': {'keyword': 'ContrastBolusIngredient', 'type': str},
            'contrast_bolus_route': {'keyword': 'ContrastBolusRoute', 'type': str},
            'contrast_bolus_volume': {'keyword': 'ContrastBolusVolume', 'type': float},
            'manufacturer': {'keyword': 'Manufacturer', 'type': str},
            'manufacturer_model_name': {'keyword': 'ManufacturerModelName', 'type': str},
            'station_name': {'keyword': 'StationName', 'type': str}
        }

        self.study_level_tags = {
            'patient_id': {'keyword': 'PatientID', 'type': str},
            'study_date': {'keyword': 'StudyDate', 'type': str},
            'accession_number': {'keyword': 'AccessionNumber', 'type': str},
            'study_description': {'keyword': 'StudyDescription', 'type': str},
        }

    @staticmethod
    def read_file(filepath, list_tags=None, stop_before_pixels=False):
        try:
            meta = pydicom.filereader.read_file(filepath,
                                                stop_before_pixels=stop_before_pixels,
                                                specific_tags=list_tags)
            return meta
        except (pydicom.errors.InvalidDicomError, IsADirectoryError):
            return None
        except OSError:
            return None

    def read_files_list(self, files_list, list_tags=None, stop_before_pixels=False):
        # Read in list of files with multithreading
        if self.num_threads > 1:
            pool = mp.Pool(self.num_threads)
            func = functools.partial(self.read_file, list_tags=list_tags, stop_before_pixels=stop_before_pixels)
            results = pool.map(func, files_list)
            pool.close()
        else:
            results = [self.read_file(f, list_tags=list_tags, stop_before_pixels=stop_before_pixels)
                       for f in files_list]

        results = [dcm for dcm in results if dcm is not None]
        return results

    # Windowing function
    def apply_window(self, image):
        range_bottom = self.win_centre - self.win_width / 2
        scale = 256 / float(self.win_width)
        image = image - range_bottom

        image = image * scale

        image[image < 0] = 0
        image[image > 255] = 255

        return image

    # Apply rescale shift
    def rescale_shift(self, image, intercept, slope):
        return image * slope + intercept

    def slice_selection(self, series_list, z_locations, slices):

        # Resize each image and stack into a np array
        resize_func = functools.partial(resize, output_shape=self.slice_selection_input_shape,
                                        preserve_range=True, anti_aliasing=True, mode='constant')
        if self.num_threads > 1:
            pool = mp.Pool(self.num_threads)
            series = pool.map(resize_func, series_list)
            series = np.dstack(series)
            pool.close()
        else:
            series = np.dstack([resize_func(im) for im in series_list])

        # Reshape the series for the network
        series = np.transpose(series[:, :, :, np.newaxis], [2, 0, 1, 3])

        # Predict the offsets for each image
        predictions = self.slice_selection_model.predict(series)
        if self.sigmoid_output:
            predictions = 2.0 * (predictions - 0.5)

        # Filter with a gaussian to smooth and take absolute value
        if self.slice_smoothing_kernel > 0.0:
            smoothing_kernel = ([self.slice_smoothing_kernel, 0.0] if self.sigmoid_output
                                else self.slice_smoothing_kernel)
            smoothed_predictions = gaussian_filter(predictions, smoothing_kernel)
        else:
            smoothed_predictions = np.copy(predictions)

        if not self.sigmoid_output:
            smoothed_predictions = np.squeeze(smoothed_predictions)

        # Post process to select a single slice
        results_by_slice = {}
        for s in slices:
            if self.sigmoid_output:
                output_index = self.slice_params[s]['slice_selection_model_output_index']
                results_by_slice[s] = self.slice_selection_post_process_sigmoids(smoothed_predictions[:, output_index],
                                                                                 z_locations)
            else:
                results_by_slice[s] = self.slice_selection_post_process(smoothed_predictions, z_locations,
                                                                        self.slice_params[s]['z_pos'])

        return results_by_slice, predictions, smoothed_predictions

    @staticmethod
    def find_zero_crossings(predictions):
        # Find zero-crossings to choose the slice
        zero_crossings = []
        for s in range(len(predictions) - 1):
            if (predictions[s] < 0.0) != (predictions[s + 1] < 0.0):
                if(abs(predictions[s]) < abs(predictions[s + 1])):
                    zero_crossings.append(s)
                else:
                    zero_crossings.append(s + 1)

        return zero_crossings


    def slice_selection_post_process_sigmoids(self, smoothed_predictions, z_locations):

        zero_crossings = self.find_zero_crossings(smoothed_predictions)

        L = len(smoothed_predictions)

        # If there are no zero-crossings, just choose the slice with the closest value to zero
        if len(zero_crossings) == 0:
            chosen_index = np.argmin(abs(smoothed_predictions))
        # Ideally there will be one zero-crossing, and this will be the chosen slice
        elif len(zero_crossings) == 1:
            chosen_index = zero_crossings[0]
        # For now just choose the first or the last, but this is quite dumb
        else:
            # Construct the ideal curve at the correct slice spacing to perform a correlation
            slice_spacing = abs(np.median(np.diff(z_locations)))
            if slice_spacing <= 0.0:
                # The slice spacings in this series are very odd
                # Fall back to simply choosing the last zero crossing
                chosen_index = zero_crossings[-1]
            else:
                ideal_array_half_length = np.floor((5.0 * self.sigmoid_scale) / slice_spacing)
                if (2 * ideal_array_half_length + 1) > L:
                    ideal_array_half_length = (L - 1) // 2
                z_lim = slice_spacing * ideal_array_half_length
                z = np.arange(-z_lim, z_lim, slice_spacing)
                ideal_curve = 2.0 * (expit(z) - 0.5)

                # Perform the correlation
                corr = np.correlate(smoothed_predictions, ideal_curve, mode='same')

                # Look for zero-crossing in the original curve that has the highest correlation value
                max_corr = -np.inf
                for x in zero_crossings:
                    if corr[x] > max_corr:
                        chosen_index = x
                        max_corr = corr[x]

        results_dict = {
            'index': int(chosen_index),
            'regression_val': float(smoothed_predictions[chosen_index]),
            'num_zero_crossings': len(zero_crossings)
        }

        return results_dict

    def slice_selection_post_process(self, smoothed_predictions, z_locations, z_pos):

        # Subtract the desired z position to get the predicted offset
        predicted_offsets = smoothed_predictions - z_pos
        L = len(smoothed_predictions)

        zero_crossings = self.find_zero_crossings(predicted_offsets)

        # If there are no zero-crossing, just choose the slice with the closest value to zero
        if len(zero_crossings) == 0:
            chosen_index = np.argmin(abs(predicted_offsets))
        # Ideally there will be one zero-crossing, and this will be the chosen slice
        elif len(zero_crossings) == 1:
            chosen_index = zero_crossings[0]
        # For now just choose the first or the last, but this is quite dumb
        else:
            # Construct the ideal curve at the correct slice spacing to perform a correlation
            slice_spacing = abs(np.median(np.diff(z_locations)))
            if slice_spacing <= 0.0:
                # The slice spacings in this series are very odd
                # Fall back to simply choosing the last zero crossing
                chosen_index = zero_crossings[-1]
            else:
                ideal_array_half_length = 10
                if (2 * ideal_array_half_length + 1) > L:
                    ideal_array_half_length = (L - 1) // 2
                z_lim = slice_spacing * ideal_array_half_length
                ideal_curve = np.linspace(-z_lim, z_lim, 2 * ideal_array_half_length + 1)

                # Check all zero crossings
                min_rms_error = np.inf
                for zc in zero_crossings:
                    # Find area of overlap between filter and signal centered at this zero-crossing
                    start_filter = max(0, -(zc - ideal_array_half_length))
                    if L - zc - 1 < ideal_array_half_length:
                        end_filter = - ideal_array_half_length + L - 1 - zc
                    else:
                        end_filter = len(ideal_curve)
                    start_signal = max(0, zc - ideal_array_half_length)
                    end_signal = min(L, zc + ideal_array_half_length + 1)

                    # Find RMS error between the signal and the ideal
                    diff = predicted_offsets[start_signal:end_signal] - ideal_curve[start_filter:end_filter]
                    rms = (diff ** 2).sum() ** 0.5
                    if rms < min_rms_error:
                        chosen_index = zc
                        min_rms_error = rms

        results_dict = {
            'index': int(chosen_index),
            'regression_val': float(smoothed_predictions[chosen_index]),
            'num_zero_crossings': len(zero_crossings)
        }

        return results_dict

    # Get DICOM metadata in a dictionary and apply normalization
    @staticmethod
    def get_dicom_metadata(dcm, tags_dict):
        results = {}
        for key, value in tags_dict.items():
            if value['keyword'] in dcm:
                tag_val = getattr(dcm, value['keyword'])
                try:
                    results[key] = value['type'](tag_val)
                except (ValueError, TypeError):
                    results[key] = None
            else:
                results[key] = None

        return results

    # Perform the segmentation on a single image
    def segmentation(self, image, slice_name):

        # Resize and reshape
        orig_shape = image.shape
        req_shape = self.slice_params[slice_name]['segmentation_input_shape']
        if image.shape != req_shape:
            image = resize(image, req_shape, preserve_range=True, anti_aliasing=True, mode='constant')
        image = np.transpose(image[:, :, np.newaxis, np.newaxis], (2, 0, 1, 3))

        segmentation_predictions = self.slice_params[slice_name]['model'].predict(image)
        segmentation_mask = np.argmax(segmentation_predictions, axis=3)
        segmentation_mask = np.squeeze(segmentation_mask)

        # Resize the segmentation mask if needed
        if segmentation_mask.shape != orig_shape:
            segmentation_mask = resize(segmentation_mask, orig_shape, order=0, preserve_range=True).astype(int)

        return segmentation_mask

    @staticmethod
    def check_boundary(mask, rec_radius_pix, num_classes):
        # Check the whether the segmentation touches the edge of the fov

        # Create an image of the radius from the centre
        m, n = mask.shape
        grid = np.mgrid[:m, :n]
        centre = np.array([[[(m - 1) / 2.0]], [[(n - 1) / 2.0]]])
        r = ((grid - centre) ** 2).sum(axis=0) ** 0.5

        # Threshold the image a few pixels inside the edge of the reconstruction area
        boundary_region = r > (rec_radius_pix - 2.0)

        # Edges of the image are always the boundary region
        boundary_region[0, :] = True
        boundary_region[-1, :] = True
        boundary_region[:, 0] = True
        boundary_region[:, -1] = True

        # Return the overlap between this boundary region and any positive class in the segmentation
        # mask
        per_class_boundary_checks = [int(np.logical_and(mask == c, boundary_region).sum())
                                     for c in range(1, num_classes + 1)]
        return per_class_boundary_checks

    @staticmethod
    def find_valid_series(datasets_list, identifier_list, min_slices=None):

        # This will contain a list of filenames for each series that passes series selection,
        # with the series instance UID string used as the key
        valid_series = defaultdict(list)

        for iden, dcm in zip(identifier_list, datasets_list):
            if (dcm is not None) and BodyCompositionEstimator.use_series(dcm):
                valid_series[dcm.SeriesInstanceUID].append(iden)

        if min_slices is not None:
            valid_series = {k: v for k, v in valid_series.items() if len(v) >= min_slices}

        return valid_series

    # Check whether a pydicom dataset belongs to a series that should be used
    @staticmethod
    def use_series(dcm):
        if 'ImageType' not in dcm:  # No image type field
            if 'ImageOrientationPatient' in dcm and dcm.ImageOrientationPatient == [1, 0, 0, 0, 1, 0]:
                return True
            else:
                return False

        # These images are dual-enegry images and are known to cause problems (not HU-based)
        if 'GSI MD' in dcm.ImageType:
            return False

        if all((kw in dcm.ImageType for kw in ['ORIGINAL', 'PRIMARY', 'AXIAL'])) or \
               (dcm.ImageType == ['DERIVED', 'PRIMARY', 'AXIAL', 'JP2K LOSSY 6:1']) or \
               (dcm.ImageType == ['DERIVED', 'SECONDARY', 'AXIAL', 'LOSSY_COMPRESSED']) or \
               (dcm.ImageType == ['ORIGINAL', 'SECONDARY', 'AXIAL']) or \
               (dcm.ImageType == ['DERIVED', 'PRIMARY', 'AXIAL', 'CT_SOM5 SPI']):
            return True

        return False

    def process_directory(self, dir_name, slices=None, save_plot='', segmentation_range=None, recursive=False,
                          return_dicom_seg=False):
        """
        Process a DICOM directory containing an entire study, select the relevant series, and process each series to
        give one body composition estimate per chosen series.

        Parameters:
        -----------
        dir_name: str
            Name of the directory containing all the image files for a single study
        slices: list of strings
            List of target slices to analyse. Each entry should match one of the keys in the slice_params dict used
            to initialise this class, e.g. ['l3', 't5', ... ]. If not specified, or set to None, every target slice
            in slice_params is analysed.
        save_plot: str
            Path in which to save a plot of the regression for each series. The .format method will be called on the
            string to insert the series instance uid within braces in the string
        segmentation_range: float
            If specified, all slices within segmentation_range mm of the chosen L3 are segemented and the results are
            averaged.
        recursive: bool
            If True, will find *.dcm files anywhere below the given directory. If false, only looks for files in the
            given directory itself.
        return_dicom_seg: bool
            Return DICOM segmentation objects in the image results

        Returns:
        --------
        dict:
            Dictionary of numerical results (json serialisable) with the following fields:
                num_valid_series: int
                    Number of series that were found to be valid and were subsequently processed
                series: dict
                    Dictionary with keys being the series instance UIDs of each series in the study that passed series
                    selection, and values being a dictionary of results for that series, as return by the
                    process_series_datasets method.

        dict:
            Dictionary of image results with the following fields:
                series: dict
                    Dictionary with keys being the series instance UIDs of each series in the study that passed series
                    selection, and values being a dictionary of image results for that series, as return by the
                    process_series_datasets method.


        """
        series_selection_tags = ['SeriesInstanceUID', 'ImageType', 'PatientID', 'AccessionNumber', 'StudyDate',
                                 'SeriesDescription', 'ImageOrientationPatient']

        # Look for files and read them in
        if recursive:
            files_list = []
            for root, _, files in os.walk(dir_name):
                for file in files:
                    full_file = os.path.join(root, file)
                    if os.path.isfile(full_file) and (file != 'DICOMDIR'):
                        files_list.append(full_file)
        else:
            files_list = [os.path.join(dir_name, f) for f in os.listdir(dir_name)]
        datasets_list = self.read_files_list(files_list, list_tags=series_selection_tags, stop_before_pixels=True)

        valid_series = self.find_valid_series(datasets_list, files_list, self.min_slices_per_series)

        # Run the full procedure on all series that passed series selection and have the required number of slices
        results = {'series': {}}
        image_results = {'series': {}}
        for series_uid, series_files_list in valid_series.items():
            series_datasets = self.read_files_list(series_files_list, stop_before_pixels=False)
            series_plot_name = save_plot.format(series_uid)
            try:
                results['series'][series_uid], image_results['series'][series_uid] = \
                    self.process_series_datasets(series_datasets,
                                                 slices,
                                                 segmentation_range=segmentation_range,
                                                 save_plot=series_plot_name,
                                                 return_dicom_seg=return_dicom_seg)
            except DICOMDecompressionError:
                # Skip this series
                continue

        # Copy over some study-level information
        results['num_valid_series'] = len(results['series'])
        results = {**results, **self.get_dicom_metadata(datasets_list[0], self.study_level_tags)}

        return results, image_results

    def process_series_files(self, dir_name, files_list, slices=None, save_plot='', gt_index=-1, gt_location=None,
                             segmentation_range=None, return_dicom_seg=False):
        """ Process a series from a list of DICOM files

        Parameters:
        -----------
        dir_name: str
            Name of the directory containing all the image files
        files_list: iterable containing strings
            An iterable (e.g. list) containing the strings that represent the file names of each image file in these
            series. The file names are relative to the 'dir_name' parameter. Note that the images in the file must
            represent a single DICOM series. However the order of the file names in the list is unimportant - they
            are sorted according to the SliceLocation tag in the header.
        slices: list of strings
            List of target slices to analyse. Each entry should match one of the keys in the slice_params dict used
            to initialise this class, e.g. ['l3', 't5', ... ]. If not specified, or set to None, every target slice
            in slice_params is analysed.
        save_plot: str
            Path in which to save a plot of the regression
        gt_index: int
            Index of the true L3 slice in the list (only needed if you want to plot the ground truth, provide either
            this or gt_location)
        gt_location: dict {str: float}
            Dictionary containing z locations of the ground truth slices in the format {slice_name: z}
            (only needed if you want to plot the ground truth)
        segmentation_range: float
            If specified, all slices within segmentation_range mm of the chosen L3 are segemented and the results are
            averaged.
        return_dicom_seg: bool
            Return DICOM segmentation objects in the image results

        Returns:
        --------
        dict
            Dictionary of results. See return value of process_series_datasets for explanataion
        """
        # Read in the list of files
        full_files_list = [os.path.join(dir_name, f) for f in files_list]
        datasets_list = self.read_files_list(full_files_list, stop_before_pixels=False)

        # Run on the list of datasets and return results
        return self.process_series_datasets(datasets_list, slices, save_plot=save_plot, gt_index=gt_index,
               gt_location=gt_location, segmentation_range=segmentation_range, return_dicom_seg=return_dicom_seg)

    def process_series_datasets(self, dataset_list, slices=None, save_plot='', gt_index=None, gt_location=None,
                                segmentation_range=None, return_dicom_seg=False):
        """ Process a series from a list of DICOM files pre-loaded into memory as pydicom datasets

        Parameters:
        -----------
        dataset_list: iterable containing pydicom datasets
            An iterable (e.g. list) containing the pre-loaded pydicom datasets to process.
            Note that the images in the dataset must represent a single DICOM series. However
            the order of the file names in the list is unimportant - they are sorted according
            to the SliceLocation tag in the header.
        slices: list of strings
            List of target slices to analyse. Each entry should match one of the keys in the slice_params dict used
            to initialise this class, e.g. ['l3', 't5', ... ]. If not specified, or set to None, every target slice
            in slice_params is analysed.
        save_plot: str
            Path in which to save a plot of the regression
        gt_index: dict {str: int}
            Dictionary containing slice indices of the ground truth slices in the list in the format
            {slice_name: index} (only needed if you want to plot the ground truth, provide either this or gt_location)
        gt_location: dict {str: float}
            Dictionary containing z locations of the ground truth slices in the format {slice_name: z}
            (only needed if you want to plot the ground truth, provide either this or gt_index)
        segmentation_range: float
            If specified, all slices within segmentation_range mm of the chosen L3 are segmented and the results are
            averaged.
        return_dicom_seg: bool
            Return DICOM segmentation objects in the image results

        Returns:
        --------
        Results are returned in two dictionaries. The first gives numerical results and is designed to be JSON
        serializable. The second contains the segmentation masks and selected images as numpy arrays.

        dict
            JSON-serializable results dictionary containing the following fields:
                num_images: int
                    Number of slices in the series
                slices: dict
                    Dictionary with one entry per target slice, using the slice name as the key. Each entry
                    is itself a dictionary with the following entries:
                        z_location: str
                            The z-location of the chosen slice in the scanner's coordinate system
                        index: int
                            Index of the chosen slice in the sorted list
                        num_zero_crossings: int
                            Number of zero crossings encountered during slice selection (if this is not 1 it could
                            indicate a problem)
                        regression_val: float
                            Value of the smoothed regression output at the chosen slice
                        sopuid: str
                            SOP Instance UID of the chosen slice
                        tissues: dict
                            Only present if segmentation_range is None. Dictionary containing per-tissue results, as
                            defined below.
                        overall: dict
                            Only present if segmentation_range is not None. A dictionary containing results that are
                            aggregated over every slice in the segmentation range. Has the following format:
                                tissues: dict
                                    Dictionary of results per tissue with the structure outlined below
                        individual: list of dict
                            Only present if segmentation_range is not None. A list of dictionaries containing results
                            for each slice within the segmentation range. Has the following format:
                                tissues: dict
                                    Dictionary of results per tissue in this slice with the structure outlined below
                                regression_val: float
                                    Value of smoothed regression output at this slice.
                                z_location: float
                                    The z-location of this slice in the scanner's coordinate system
                                index: int
                                    Index of this slice in the sorted list
                                sopuid: str
                                    SOP Instance UID of the this slice
                                offset_from_chosen: float
                                    Signed distance of this slice from the primary chosen slice

        dict
            Dictionary containing image results. There is one entry per slice ()
                seg_mask: np.array (or list of np.arrays if a segmentation_range is specified)
                    The segmentation mask (or list of masks if segmentation_range is specified)
                image: np.array (or list of np.arrays if a segmentation_range is specified)
                    The (unwindowed) image that analysis was performed on (or list of images if segmentation_range
                    is specified)

        Where a dictionary of tissue results is referenced above, it has the following format:
        tissues: dictionary
            Dictionary with one entry for each tissue type present in this target slice, e.g. 'muscle',
            each of which is itself a dictionary containing the following fields:
                area_cm2: float
                    Area (in cm2) of the tissue types
                mean_hu: float
                    Mean Hounsfield unit within the segmented area of the this tissue
                median_hu: float
                    Median Hounsfield unit within the segmented area of the this tissue
                std_hu: float
                    Standard deviation of Hounsfield units within the segmented area of this tissue
                iqr_hu: float
                    Interquartile range of Hounsfield units within the segmented area of the this tissue
                boundary_check: int
                    Number of pixels intersection between the segmentation mask for this class and the edge of the
                    reconstructed area
        """
        # Ensure all images are from the same series types
        series_uid_list = [dcm.SeriesInstanceUID for dcm in dataset_list]
        if not all([uid == series_uid_list[0] for uid in series_uid_list[1:]]):
            raise RuntimeError('The files requested do not belong to the same series')

        # Check that the requested slices are understood
        if slices is None:
            slices = list(self.slice_params.keys())
        else:
            for s in slices:
                if s not in self.slice_params:
                    raise ValueError("Unrecognised slice '{}', recognised values are: [{}]"
                                     .format(s, ', '.join(self.slice_params.keys())))

        # Get the ground_truth z location (if needed)
        if gt_location is not None and gt_location is not None:
            raise ValueError("Provide either gt_location or gt_index, but not both")
        if gt_location is None and gt_index is not None:
            gt_location = {s: float(dataset_list[gti].ImagePositionPatient[2]) for s, gti in gt_location.items()}

        # Sort the datasets by physical location in the z direction
        dataset_list = sorted(dataset_list, key=lambda dcm: float(dcm.ImagePositionPatient[2]))

        # Image processing
        images_list = []
        windowed_images = []
        z_locations = []
        for dcm in dataset_list:
            try:
                image = dcm.pixel_array
            except UnicodeEncodeError:
                continue
            slope = float(dcm.RescaleSlope)
            intercept = float(dcm.RescaleIntercept)
            image = self.rescale_shift(image, intercept, slope)

            # Sometimes pydicom decmpresses images incorrectly, which is fortunately easy to detect
            # because the range of values will be totally off. This could also be caused by artefects,
            # but these should also be rejected
            if (image.min() < -11000) or (image.max() > 12000):
                raise DICOMDecompressionError(("Unexpected HU range indicates presence of decompression error "
                                               "or artefact"))

            # Rotate image to supine position if not in supine
            patient_position = dcm.PatientPosition
            if patient_position.endswith('P'):  # prone
                image = np.rot90(image, k=2)
            elif patient_position.endswith('DL'):  # decubitus left
                image = np.rot90(image, k=1)
            elif patient_position.endswith('DR'):  # decubitus right
                image = np.rot90(image, k=3)

            images_list.append(image)

            # Windowing preprocessing
            windowed_images.append(self.apply_window(image))

            z_locations.append(float(dcm.ImagePositionPatient[2]))

        # Results dictionaries that will be returned
        results = {}
        image_results = {}

        # Get some series-level information from the first slice
        results = {**results, **self.get_dicom_metadata(dataset_list[0], self.series_level_tags)}

        # Occasionally series descriptions are missing, or in very odd situations, are lists
        series_description = dataset_list[0].SeriesDescription if "SeriesDescription" in dataset_list[0] else None
        if isinstance(series_description, pydicom.multival.MultiValue):
            series_description = ' '.join(list(series_description))
        results['series_description'] = series_description

        # Perform slice selection
        results['slices'], raw_predictions, smoothed_predictions = self.slice_selection(windowed_images,
                                                                                        z_locations,
                                                                                        slices)

        # Iterate through the requested levels
        for s in slices:
            chosen_image_index = results['slices'][s]['index']
            results['slices'][s]['sopuid'] = dataset_list[chosen_image_index].SOPInstanceUID
            results['slices'][s]['z_location'] = z_locations[chosen_image_index]
            num_classes = len(self.slice_params[s]['class_names'])

            image_results[s] = {}

            # Perform segmentation
            if segmentation_range is not None:
                # Create a list to contain results for each slice within the range
                results['slices'][s]['individual'] = []
                results['slices'][s]['overall'] = {}
                results['slices'][s]['overall']['tissues'] = {}

                image_results[s]['seg_mask'] = []
                image_results[s]['image'] = []

                chosen_z = z_locations[chosen_image_index]

                # Checks on the boundary region that will accumulate over the entire region
                overall_boundary_checks = [None] * num_classes

                chosen_datasets = []

                combined_seg_values = {tis: [] for tis in self.slice_params[s]['class_names']}

                # Initialise the 'overall' results
                for tis in self.slice_params[s]['class_names']:
                    results['slices'][s]['overall']['tissues'][tis] = {}
                    results['slices'][s]['overall']['tissues'][tis]['area_cm2'] = 0.0

                # Loop through slices
                for ind, (dcm, z, im, wim) in enumerate(zip(dataset_list, z_locations, images_list, windowed_images)):
                    if abs(z - chosen_z) <= segmentation_range:
                        # Results dictionary for this individual slice
                        slice_results = {}
                        slice_results['z_location'] = z
                        slice_results['index'] = ind
                        slice_results['sopuid'] = dcm.SOPInstanceUID
                        slice_results['offset_from_chosen'] = z - chosen_z
                        slice_results['tissues'] = {tis: {} for tis in self.slice_params[s]['class_names']}
                        chosen_datasets.append(dcm)

                        if self.sigmoid_output:
                            output_index = self.slice_params[s]['slice_selection_model_output_index']
                            slice_results['regression_val'] = float(smoothed_predictions[ind, output_index])
                        else:
                            slice_results['regression_val'] = float(smoothed_predictions[ind])

                        # Find the area of a single pixel
                        pixel_spacing = dcm.PixelSpacing
                        pixel_area = float(pixel_spacing[0]) * float(pixel_spacing[1]) / 100.0

                        # Run segmentation
                        mask = self.segmentation(wim, s)
                        image_results[s]['seg_mask'].append(mask)
                        image_results[s]['image'].append(im)

                        # Find the reconstruction radius in pixel units and use it to perform a boundary check if it's
                        # there
                        if "ReconstructionDiameter" in dcm:
                            rec_diameter = float(dcm.ReconstructionDiameter)
                            rec_radius_pix = 0.5 * (rec_diameter / float(pixel_spacing[0]))
                            boundary_checks = self.check_boundary(mask, rec_radius_pix, num_classes)
                            # Update the total boundary_checks
                            for cind, bc in enumerate(boundary_checks):
                                if overall_boundary_checks[cind] is None:
                                    overall_boundary_checks[cind] = bc
                                else:
                                    overall_boundary_checks[cind] += bc
                        else:
                            boundary_checks = [None] * num_classes

                        # Store the per-slice boundary check results
                        for tis, bc in zip(self.slice_params[s]['class_names'], boundary_checks):
                            slice_results['tissues'][tis]['boundary_check'] = bc

                        # Add slice level DICOM metadata
                        slice_results = {**slice_results,
                                         **self.get_dicom_metadata(dcm, self.instance_level_tags)}

                        # Results for each tissue
                        for i, tis in enumerate(self.slice_params[s]['class_names']):
                            c = i + 1  # offset of one due to background class
                            pixel_count = (mask == c).sum()
                            area = float(pixel_count * pixel_area)

                            slice_results['tissues'][tis]['area_cm2'] = area
                            results['slices'][s]['overall']['tissues'][tis]['area_cm2'] += area
                            seg_values = im[mask == c]
                            combined_seg_values[tis].append(seg_values)
                            slice_results['tissues'][tis] = {**slice_results['tissues'][tis],
                                                             **self.find_pixel_statistics(seg_values)}
                        results['slices'][s]['individual'].append(slice_results)

                # Find aggregate 'overall' results by combining results from different slices
                for tis, bc in zip(self.slice_params[s]['class_names'], overall_boundary_checks):
                    results['slices'][s]['overall']['tissues'][tis]['area_cm2'] /= \
                        len(results['slices'][s]['individual'])
                    results['slices'][s]['overall']['tissues'][tis]['boundary_check'] = \
                        int(bc) if bc is not None else None
                    if len(combined_seg_values[tis]) > 0:
                        combined_seg_values_arr = np.hstack(combined_seg_values[tis])
                        results['slices'][s]['overall']['tissues'][tis] = {
                            **self.find_pixel_statistics(combined_seg_values_arr),
                            **results['slices'][s]['overall']['tissues'][tis]
                        }
                    else:
                        for key in ['mean_hu', 'median_hu', 'std_hu', 'iqr_hu']:
                            results['slices'][s]['overall']['tissues'][tis][key] = np.nan

                if return_dicom_seg:
                    image_results[s]['dicom_seg'] = self.make_dicom_seg(
                                                        image_results[s]['seg_mask'],
                                                        chosen_datasets,
                                                        self.slice_params[s]['class_names']
                                                    )

            else:
                dcm = dataset_list[chosen_image_index]

                # Perform segmentation
                seg_mask = self.segmentation(windowed_images[chosen_image_index], s)

                # Store results
                image_results[s]['seg_mask'] = seg_mask
                image_results[s]['image'] = images_list[chosen_image_index]
                if return_dicom_seg:
                    image_results[s]['dicom_seg'] = self.make_dicom_seg(
                        [seg_mask],
                        [dcm],
                        self.slice_params[s]['class_names']
                    )

                pixel_spacing = dcm.PixelSpacing
                pixel_area = float(pixel_spacing[0]) * float(pixel_spacing[1]) / 100.0

                # Find the reconstruction radius in pixel units and use it to perform a boundary check if it's there
                if "ReconstructionDiameter" in dcm:
                    rec_diameter = float(dcm.ReconstructionDiameter)
                    rec_radius_pix = 0.5 * (rec_diameter / float(pixel_spacing[0]))
                    boundary_checks = self.check_boundary(seg_mask, rec_radius_pix, num_classes)
                else:
                    boundary_checks = [None] * num_classes

                # Loop over tissues and calculate metrics
                results['slices'][s]['tissues'] = {}
                for i, (tis, bc) in enumerate(zip(self.slice_params[s]['class_names'], boundary_checks)):
                    c = i + 1  # offset of one due to background class
                    results['slices'][s]['tissues'][tis] = {}
                    pixel_count = (seg_mask == c).sum()
                    results['slices'][s]['tissues'][tis]['area_cm2'] = float(pixel_count * pixel_area)
                    results['slices'][s]['tissues'][tis]['boundary_check'] = bc
                    seg_values = images_list[chosen_image_index][seg_mask == c]
                    if len(seg_values) == 0:
                        for key in ['mean_hu', 'median_hu', 'std_hu', 'iqr_hu']:
                            results['slices'][s]['tissues'][tis][key] = np.nan
                    else:
                        # Add pixel statistics to the result
                        results['slices'][s]['tissues'][tis] = {**self.find_pixel_statistics(seg_values),
                                                                **results['slices'][s]['tissues'][tis]}

                # Add slice level DICOM metadata
                results['slices'][s] = {**results['slices'][s],
                                        **self.get_dicom_metadata(dcm, self.instance_level_tags)}

        # Save a plot
        if len(save_plot) > 0:
            chosen_indices = {s: results['slices'][s]['index'] for s in slices}
            self.plot_regression(z_locations, raw_predictions, smoothed_predictions, chosen_indices,
                                 save_plot, gt_location=gt_location)

        results['num_images'] = len(images_list)
        return results, image_results

    @staticmethod
    def find_pixel_statistics(pixels):
        results = {}
        if len(pixels) < 1:
            results['mean_hu'] = None
            results['median_hu'] = None
            results['std_hu'] = None
            results['iqr_hu'] = None
        else:
            results['mean_hu'] = float(pixels.mean())
            results['median_hu'] = float(np.median(pixels))
            results['std_hu'] = float(np.std(pixels))
            q75, q25 = np.percentile(pixels, [75, 25])
            results['iqr_hu'] = float(q75 - q25)
        return results

    def plot_regression(self, z_locations, predictions, smoothed_predictions, chosen_index,
                        file_name, gt_location=None):

        fig = plt.figure(figsize=(4, 4))
        z_locations = np.array(z_locations)
        if self.sigmoid_output:
            for s in self.slice_params.keys():
                index = self.slice_params[s]['slice_selection_model_output_index']
                colour = self.slice_params[s]['regression_plot_colour']
                plt.plot(z_locations, predictions[:, index], c=colour, linestyle=':', label=s.upper() + ' Prediction')
                plt.plot(z_locations, smoothed_predictions[:, index], c=colour,
                         label=s.upper() + ' Smoothed Prediction')
        else:
            plt.plot(z_locations, predictions, c='gray', linestyle=':', label='Prediction')
            plt.plot(z_locations, smoothed_predictions, c='gray', label='Smoothed Prediction')
        for s, ind in chosen_index.items():
            colour = self.slice_params[s]['regression_plot_colour']
            plt.axvline(x=z_locations[ind], c=colour, linestyle='-', linewidth=0.3)
        if gt_location is not None:
            for s, zloc in gt_location.items():
                colour = self.slice_params[s]['regression_plot_colour']
                plt.axvline(x=zloc, c=colour, linestyle='-.', linewidth=0.3)

        plt.legend()
        plt.grid(True)
        plt.ylabel('Prediction')
        plt.xlabel('Z-axis (mm)')
        fig.savefig(file_name, dpi=512 / 4)
        plt.close(fig)

    def make_dicom_seg(self, mask_list, dcm_list, class_names):

        if len(mask_list) > 1:
            mask = np.stack(mask_list, axis=0).astype(np.uint8)
        else:
            mask = mask_list[0].astype(np.uint8)

        # Describe the algorithm that created the segmentation
        algorithm_identification = AlgorithmIdentificationSequence(
            name=MODEL_NAME,
            version=self.algorithm_version,
            family=codes.cid7162.ArtificialIntelligence
        )

        # Check that we have descriptions for all the segments
        for c in class_names:
            if c not in KNOWN_SEGMENT_DESCRIPTIONS:
                raise KeyError(f'There is no known segment description for class {c}')

        # Describe the segment
        segment_descriptions = [
            SegmentDescription(
                segment_number=i,
                segment_label=KNOWN_SEGMENT_DESCRIPTIONS[tis]['segment_label'],
                segmented_property_category=KNOWN_SEGMENT_DESCRIPTIONS[tis]['segmented_property_category'],
                segmented_property_type=KNOWN_SEGMENT_DESCRIPTIONS[tis]['segmented_property_type'],
                algorithm_type=SegmentAlgorithmTypeValues.AUTOMATIC,
                algorithm_identification=algorithm_identification,
            )
            for i, tis in enumerate(class_names, start=1)
        ]

        # Create the Segmentation instance
        seg_dataset = Segmentation(
            source_images=dcm_list,
            pixel_array=mask,
            segmentation_type=SegmentationTypeValues.BINARY,
            segment_descriptions=segment_descriptions,
            series_instance_uid=pydicom.uid.generate_uid(),
            series_number=100,
            sop_instance_uid=pydicom.uid.generate_uid(),
            instance_number=1,
            manufacturer=MANUFACTURER,
            manufacturer_model_name=MODEL_NAME,
            software_versions=self.algorithm_version,
            transfer_syntax_uid=ExplicitVRLittleEndian,
            device_serial_number=SERIAL_NUMBER,
        )

        return seg_dataset
