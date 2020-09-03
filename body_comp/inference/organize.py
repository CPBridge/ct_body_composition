import os
from pathlib import Path
import pydicom


def organize_data(top: Path, new_top: Path):
    """Create a re-organized copy of all the DICOM files underneath one directory into mrn_acc format.

    This uses the meta-data in the DICOM files themselves.

    Parameters
    ----------
    top: Path
        Directory to be organized
    new_top: Path
        Directory to place new data in

    """

    # Check the input directory exists
    if not top.exists():
        raise FileNotFoundError('The top level directory does not exist')

    # Create the output directory if it doesn't exist
    if not new_top.exists():
        new_top.mkdir()

    # Walk the directory tree
    for root, _, files in os.walk(str(top)):
        for f in files:
            print(os.path.join(root, f))
            try:
                dcm = pydicom.read_file(os.path.join(root, f))
            except pydicom.errors.InvalidDicomError:
                print("Invalid DICOM", f)
                continue

            mrn = dcm.PatientID
            acc = dcm.AccessionNumber.replace('.', '').replace(' ', '')
            if len(acc.replace(' ', '')) == 0:
                acc = 'unknown'
            out_dir = new_top / '{}_{}'.format(mrn, acc)
            out_dir.mkdir(exist_ok=True)
            out = out_dir / f
            print(out)
            dcm.save_as(str(out))
