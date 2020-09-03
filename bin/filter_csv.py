#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path

from body_comp.inference.utils import filter_csv


if __name__ == '__main__':
    parser = ArgumentParser(description='Filter a full raw results csv to remove series and implement checks')
    parser.add_argument('input_csv', type=Path, help='Raw results csv')
    parser.add_argument('output_csv', type=Path, help='Name of output file to create')
    parser.add_argument('--slices', nargs='+', default=('L3', ),
                        help='Which slices to filter on')
    parser.add_argument('--boundary_checks', action='store_true', help='Apply boundary check filtering')

    filter_csv(**vars(parser.parse_args()))
