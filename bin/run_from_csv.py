#!/usr/bin/env python3
import argparse

from body_comp.inference.utils import run_body_comp_csv


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=
                            'Run the body composition algorithm on a cohort of patients in a csv file')
    parser.add_argument('in_csv', help='Input csv listing the studies to analyse')
    parser.add_argument('estimator_config', help='A json file containing parameters for the estimator')
    parser.add_argument('output_dir', help='Output directory for image results. Should be empty unless keep_existing '
                                           'is specified.')
    parser.add_argument('input_dirs', nargs='+', help='Directory(ies) containing input files')
    parser.add_argument('--num_threads', '-t', type=int, default=1, help='Number of threads to use')
    parser.add_argument('--segmentation_range', '-r', type=float,
                        help='Segment all slices with this range (leave unspecified for single slice)')
    parser.add_argument('--keep_existing', '-k', action='store_true',
                        help='Skip studies that already have a file in the output directory')
    parser.add_argument('--recursive', '-R', action='store_true',
                        help='Look for images recursively')
    parser.add_argument('--rerun_exceptions', '-e', action='store_true',
                        help='Re-try any studies that encountered an exception in the previous run. '
                             'Only valid if keep_existing is also enabled.')
    parser.add_argument('--use_directory_list', '-l', action='store_true',
                        help='Treat the input directory as a text file containing directories')
    parser.add_argument('--dicom_seg', '-d', action='store_true',
                        help='Save dicom seg files')
    parser.add_argument('--min_slices_per_series', '-m', type=int, default=20,
                        help='Reject series with fewer than this number of instances')
    args = parser.parse_args()

    run_body_comp_csv(**vars(args))
