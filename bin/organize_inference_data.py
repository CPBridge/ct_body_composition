#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path

from body_comp.inference.organize import organize_data


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Create a re-organized copy of a directory tree of DICOM files with any structure MRN_ACC'
                    '. The data copy is in directories named with the mrn_acc convention.'
    )
    parser.add_argument('top', type=Path, help='Top directory of the existing file structure')
    parser.add_argument('new_top', type=Path, help='Top directory of the new file structure to be created')
    organize_data(**vars(parser.parse_args()))
