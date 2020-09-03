import os
from pathlib import Path
import json
from tempfile import TemporaryDirectory

import numpy as np

import pandas as pd

from imageio import imsave, imread

from skimage.color import gray2rgb
from skimage.color.colorlabel import label2rgb
from skimage.transform import resize

from body_comp.inference.estimator import BodyCompositionEstimator


# Colours of the mask for the compartments
MASK_COLOURS = (
    'black',    # background
    'red',      # muscle
    'green',    # visceral fat
    'yellow'    # subcutaneous_fat
)

# Columns in the output csv file
OUTPUT_COLUMN_ORDER = ('Project', 'Cohort', 'masterExamIdentifier', 'EMPI', 'Location', 'MRN', 'ACC', 'DateExam',
                       'StudyName', 'FoundImageData', 'ImageDataLocation', 'NumSeriesSelected', 'ExceptionEncountered',
                       'ExceptionMessage')

# Limits used for the boundary checks of each component
# (number of pixels close to reconstruction boundary)
BOUNDARY_CHECKS = {
    'subcutaneous_fat': 100,
    'visceral_fat': 0,
    'muscle': 0
}


def vert_location(v):
    """Map a vertebra name to a location down the spine."""
    return {'C': 0, 'T': 7, 'L': 19, 'S': 24}[v[0].lower()] + int(v[1])


def save_results_csv(study_summary_list, out_csv):
    # Save the results
    columns = [c for c in OUTPUT_COLUMN_ORDER if c in study_summary_list[0].keys()]
    df = pd.DataFrame(study_summary_list, columns=columns)
    df.to_csv(out_csv)


def save_image_results(study_name, study_results, image_results, output_plot, preview_output_dir, all_slices_output_dir,
                       estimator, segmentation_range):

    # Store images
    for series_uid, series_results in study_results['series'].items():

        # Create a preview image containing the central image and the regression plot
        reg_file = output_plot.format(series_uid)
        reg_image = imread(reg_file)
        reg_image = reg_image[:, :, :3]
        os.remove(reg_file)

        # Place blank space next to the regression image
        reg_image_padded = np.hstack([reg_image, 255 * np.ones_like(reg_image)])
        preview_panels = [reg_image_padded]

        slices = list(series_results['slices'].keys())
        try:
            slices = sorted(slices, key=vert_location)
        except:
            pass
        for s in slices:
            if segmentation_range is None:
                mask = image_results['series'][series_uid][s]['seg_mask']
                image = image_results['series'][series_uid][s]['image']
            else:
                chosen_slice_uid = series_results['slices'][s]['sopuid']
                # Need to find the correct element of the list for the chosen slice
                for slice_ind, slice_data in enumerate(series_results['slices'][s]['individual']):
                    if slice_data['sopuid'] == chosen_slice_uid:
                        list_index = slice_ind
                        break
                mask = image_results['series'][series_uid][s]['seg_mask'][list_index]
                image = image_results['series'][series_uid][s]['image'][list_index]
            image = estimator.apply_window(image)
            if image.shape != reg_image.shape[:2]:
                image = resize(image, reg_image.shape[:2], preserve_range=True, clip=False)
            if mask.shape != reg_image.shape[:2]:
                mask = resize(mask, reg_image.shape[:2], preserve_range=True, clip=False, order=0)
            colour_mask = (label2rgb(mask, colors=MASK_COLOURS, bg_label=-1) * 255).astype(np.uint8)
            colour_image = gray2rgb(image).astype(np.uint8)
            output_image = np.hstack([colour_image, colour_mask])
            preview_panels.append(output_image)

        # Stack the panels for each slice
        preview_image_output = np.vstack(preview_panels).astype(np.uint8)
        image_path = os.path.join(preview_output_dir, '{}_{}_preview.png'.format(study_name, series_uid))
        imsave(image_path, preview_image_output)

        # Store all images and masks when multislice analysis was used
        if segmentation_range is not None:

            # Create a new subdirectory to hold images for this series
            series_output_dir = os.path.join(all_slices_output_dir, '{}_{}'.format(study_name, series_uid))
            os.makedirs(series_output_dir, exist_ok=True)

            for s in series_results['slices'].keys():

                masks_list = image_results['series'][series_uid][s]['seg_mask']
                images_list = image_results['series'][series_uid][s]['image']

                for j, (im, mask) in enumerate(zip(images_list, masks_list)):

                    # Change the mask to colour and chosen image to RGB
                    mask = (label2rgb(mask, colors=MASK_COLOURS, bg_label=-1) * 255).astype(np.uint8)

                    im = estimator.apply_window(im)
                    im = gray2rgb(im).astype(np.uint8)

                    composite_image = np.hstack([im, mask]).astype(np.uint8)

                    image_path = os.path.join(series_output_dir, '{}_{}.png'.format(s, j))
                    imsave(image_path, composite_image)


def run_body_comp_csv(in_csv, input_dirs, output_dir, estimator_config=None, segmentation_range=None, dicom_seg=False,
                      keep_existing=False, use_directory_list=False, num_threads=10, rerun_exceptions=False,
                      recursive=False, min_slices_per_series=20):

    if rerun_exceptions and not keep_existing:
        raise ValueError('Enabling rerun_exceptions is not valid if keep_existing is not enabled')

    # If using a directory list file, read it in
    if use_directory_list:
        if len(input_dirs) != 1:
            raise ValueError('If use_directory_list, specify a single file')
        with open(input_dirs[0], 'r') as dir_file:
            input_dirs = [d.rstrip('\n') for d in dir_file.readlines()]

    # Load in config file
    if estimator_config is None:
        estimator_config = {}
    elif isinstance(estimator_config, str) or isinstance(estimator_config, Path):
        with open(str(estimator_config), 'r') as jsonfile:
            estimator_config = json.load(jsonfile)

    # Set up the model object
    estimator = BodyCompositionEstimator(**estimator_config,
                                         num_threads=num_threads,
                                         min_slices_per_series=min_slices_per_series)

    # Make sure the output directories exist and are empty
    if os.path.exists(output_dir) and not keep_existing:
        if len(os.listdir(output_dir)) > 0:
            raise RuntimeError('Output directory is not empty. Exiting.')
    os.makedirs(output_dir, exist_ok=True)

    json_output_dir = os.path.join(output_dir, 'json_files')
    os.makedirs(json_output_dir, exist_ok=True)
    preview_output_dir = os.path.join(output_dir, 'previews')
    os.makedirs(preview_output_dir, exist_ok=True)
    if segmentation_range is not None:
        all_slices_output_dir = os.path.join(output_dir, 'all_slices')
        os.makedirs(all_slices_output_dir, exist_ok=True)
    else:
        all_slices_output_dir = None
    if dicom_seg:
        seg_output_dir = os.path.join(output_dir, 'dicom_seg')
        os.makedirs(seg_output_dir, exist_ok=True)

    # Results file goes in the results directory
    out_csv = os.path.join(output_dir, 'run_log.csv')

    # Read in the input list of studies
    in_df = pd.read_csv(in_csv, dtype=str)
    if 'StudyName' in in_df.columns:
        use_study_name = True
    elif 'MRN' in in_df.columns and 'ACC' in in_df.columns:
        use_study_name = False
    else:
        raise RuntimeError(
            'The input CSV must contain either a StudyName column, or MRN and ACC columns'
        )

    # Drop any of those annoying Unnamed columns that have crept in
    for c in in_df.columns:
        if 'Unnamed' in c:
            in_df.drop(c, inplace=True, axis=1)

    # Convert dict to records to iterate through
    in_dicts = in_df.to_dict('records')

    # Get a list of studies that have already been processed
    if keep_existing:
        out_df = pd.read_csv(
            out_csv,
            index_col=0,
            dtype={k : str for k in ['MRN', 'ACC', 'EMPI', 'masterExamIdentifier', 'StudyName']}
        )

        # Remove the studies that previously encountered exceptions so that they will be re-run
        if rerun_exceptions:
            out_df = out_df[~out_df.ExceptionEncountered].copy()

        study_summary_list = out_df.to_dict('records')
        if use_study_name:
            existing_studies = set([row['StudyName'] for row in study_summary_list])
        else:
            existing_studies = set([row['MRN'] + '_' + row['ACC'] for row in study_summary_list])
    else:
        study_summary_list = []

    # Loop over studies
    for i, study_dict in enumerate(in_dicts):

        if use_study_name:
            study_name = study_dict['StudyName']
        else:
            study_name = study_dict['MRN'] + '_' + study_dict['ACC']

        # Skip this study if there are already results for it in the output directory
        if keep_existing and study_name in existing_studies:
            continue
        print('{}/{}'.format(i, len(in_dicts)), study_name)

        study_summary = study_dict.copy()

        candidate_study_dirs = []
        for input_dir in input_dirs:
            study_dir = os.path.join(input_dir, study_name)
            if os.path.exists(study_dir):
                candidate_study_dirs.append(study_dir)

        if len(candidate_study_dirs) < 1:
            study_summary['FoundImageData'] = False
            study_summary['ImageDataLocation'] = ''
            study_summary['ExceptionEncountered'] = False
            study_summary['ExceptionMessage'] = ''
            study_summary['NumSeriesSelected'] = 0
            study_summary_list.append(study_summary)
            save_results_csv(study_summary_list, out_csv)
            continue
        else:
            study_summary['FoundImageData'] = True

        # Create a temporary directory to store the regression plot before it is stitched into the composite
        with TemporaryDirectory() as intermediate_output_dir:
            output_plot = os.path.join(intermediate_output_dir, study_name + '_{}_regression.png')

            for study_dir in candidate_study_dirs:
                study_summary['ImageDataLocation'] = study_dir

                try:
                    study_results, image_results = estimator.process_directory(dir_name=study_dir,
                                                                               save_plot=output_plot,
                                                                               segmentation_range=segmentation_range,
                                                                               return_dicom_seg=dicom_seg,
                                                                               recursive=recursive)
                except Exception as e:
                    print(e)
                    study_summary['ExceptionEncountered'] = True
                    study_summary['ExceptionMessage'] = str(e)
                    study_summary['NumSeriesSelected'] = 0
                    continue

                study_summary['ExceptionEncountered'] = False
                study_summary['ExceptionMessage'] = ''
                study_summary['NumSeriesSelected'] = study_results['num_valid_series']

                if study_results['num_valid_series'] > 0:
                    break
            else:
                # No candidate study dir succeeded
                study_summary_list.append(study_summary)
                save_results_csv(study_summary_list, out_csv)
                continue

            # Store numerical results as a JSON file
            study_results = {**study_results, **study_summary}
            results_file = os.path.join(json_output_dir, study_name + '.json')
            with open(results_file, 'w') as jsonfile:
                json.dump(study_results, jsonfile, indent=2)

            save_image_results(study_name=study_name,
                               study_results=study_results,
                               image_results=image_results,
                               output_plot=output_plot,
                               preview_output_dir=preview_output_dir,
                               all_slices_output_dir=all_slices_output_dir,
                               estimator=estimator,
                               segmentation_range=segmentation_range)

            if dicom_seg:
                for series_uid, series_results in image_results['series'].items():
                    for slice_name, slice_results in series_results.items():
                        seg_file = os.path.join(seg_output_dir, '{}_{}_{}'.format(study_name, series_uid, slice_name))
                        slice_results['dicom_seg'].save_as(seg_file)

        # Store results
        study_summary_list.append(study_summary)
        save_results_csv(study_summary_list, out_csv)

    # Create a summary csv file
    summary_file_path = os.path.join(output_dir, 'summary.csv')
    filtered_summary_file_path = os.path.join(output_dir, 'filtered_summary.csv')
    gather_results_to_csv(json_output_dir, summary_file_path, multislice=segmentation_range is not None)
    slice_list = list(estimator.slice_params.keys())
    filter_csv(summary_file_path, filtered_summary_file_path, slices=slice_list)


def gather_results_to_csv(input_dir, output_file, multislice=False, center=False):

    input_dir = Path(input_dir)

    # Loop over .json files in the specified directory
    output_list = []
    for f in input_dir.glob('*.json'):

        study_name = f.name.replace('.json', '')

        # Load in the data in the json file
        with f.open('r') as jf:
            results_data = json.load(jf)

            # Loop over the series in this study
            for series_uid, series_data in results_data['series'].items():

                series_results = {'study_name': study_name}
                series_results['series_instance_uid'] = series_uid
                # Study level information
                for el in ['num_valid_series', 'study_description', 'study_date', 'patient_id', 'accession_number']:
                    if el in results_data:
                        series_results[el] = results_data[el]
                # Series level information
                for el in ['series_description', 'slice_thickness_mm', 'manufacturer',
                           'manufacturer_model_name', 'station_name', 'num_images']:
                    if el in series_data:
                        series_results[el] = series_data[el]

                for slice_name, slice_data in series_data['slices'].items():

                    series_results['{}_zero_crossings'.format(slice_name)] = slice_data['num_zero_crossings']
                    series_results['{}_regression_val'.format(slice_name)] = slice_data['regression_val']
                    series_results['{}_sop_instance_uid'.format(slice_name)] = slice_data['sopuid']
                    series_results['{}_z_location'.format(slice_name)] = slice_data['z_location']

                    # Where to find the data to extract to the CSV now depends on whether multislice analysis was run
                    if multislice:
                        # Use only the metadata and, if specified, results from the center slice
                        for s in slice_data['individual']:
                            if s['offset_from_chosen'] == 0.0:
                                if center:
                                    tissue_items = s['tissues'].items()
                                # Copy over instance-level metadata
                                for el in ['exposure_mAs', 'exposure_time_ms', 'tube_current_mA', 'kvp']:
                                    if el in s:
                                        series_results['{}_{}'.format(slice_name, el)] = s[el]
                                break
                        else:
                            print('Warning: no center slice found for {} in study {} series {}!'.format(slice_name,
                                                                                                        study_name,
                                                                                                        series_uid))
                            continue
                        if not center:
                            # Default for multislice anlysis is to use the overall values aggregated from all the slices
                            tissue_items = slice_data['overall']['tissues'].items()
                    else:
                        # Copy over instance-level metadata
                        for el in ['exposure_mAs', 'exposure_time_ms', 'tube_current_mA', 'kvp']:
                            if el in slice_data:
                                series_results['{}_{}'.format(slice_name, el)] = slice_data[el]

                        # Simple single slice anaysis - use results from the single slice
                        tissue_items = slice_data['tissues'].items()
                    for tissue_name, tissue_data in tissue_items:

                        for prop in ['median_hu', 'std_hu', 'mean_hu', 'area_cm2', 'iqr_hu', 'boundary_check']:
                            series_results['{}_{}_{}'.format(slice_name, tissue_name, prop)] = tissue_data[prop]

                output_list.append(series_results)

    pd.DataFrame(output_list).to_csv(output_file)


def is_uid(col_name: str) -> bool:
    '''Determine whether a column name from a CSV file represents a UID

    Parameters
    ----------
    col_name: str
        Name of the column

    Returns
    -------
    bool
        True if and only if the column name represents a uid

    '''
    if col_name in ('series_instance_uid', 'study_instance_uid'):
        return True
    if 'sop_instance_uid' in col_name:
        return True
    return False


def filter_csv(input_csv: Path, output_csv: Path, choose_thickest: bool = True, slices=['L3'],
               boundary_checks: bool = False):

    # Open the input csv with no rows to get the columns names
    df = pd.read_csv(str(input_csv), index_col=0, nrows=0)

    # Ensure any column containing a UID is read in as a string
    column_types = {c: str for c in df.columns if is_uid(c)}

    # Open the input csv
    df = pd.read_csv(str(input_csv), index_col=0, dtype=column_types)
    initial_len = len(df)
    initial_n_studies = df.study_name.nunique()
    print("initial studies", initial_n_studies)

    # Initially include all series, then filter out unwanted ones
    ind = pd.Series(True, index=df.index)

    for slice_name in slices:
        # Filter series with multiple zero-crossings or no zero-crossings
        ind &= df['{}_zero_crossings'.format(slice_name)] == 1

        # Check how many studies were rejected based on slice selection
        n_studies_after_slice_selection = df[ind].study_name.nunique()
        print(initial_n_studies - n_studies_after_slice_selection, "studies dropped due to slice selection")

        if boundary_checks:
            # Filter based on boundary checks
            for tis, val in BOUNDARY_CHECKS.items():
                ind &= df['{}_{}_boundary_check'.format(slice_name, tis)] <= val

            # Check number of studies after boundary checks
            n_studies_after_boundary_check = df[ind].study_name.nunique()
            print(n_studies_after_slice_selection - n_studies_after_boundary_check,
                  "further dropped after boundary checks")

    # Apply the index
    df = df[ind].copy()

    if choose_thickest:
        # Sort by slice thickness within a given study and then drop thinner sliced studies, smaller series and
        # slices with lower visceral fat amounts at L3
        df = df.sort_values(
            by=['study_name', 'slice_thickness_mm', 'num_images', 'L3_visceral_fat_area_cm2']
        ).drop_duplicates(subset=['study_name'], keep='last').copy()

    # Store to file
    print('{} of {} initial series retained'.format(len(df), initial_len))
    df.to_csv(str(output_csv))
