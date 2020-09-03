# Direct Installation

Part of this repository contains a Python package called `body_comp`, which should be installed before running any
of the code.
Once you have cloned this repository you can install the `body_comp` package and all its dependencies by running
this command from the root of the repository:

```bash
$ pip install .
```

Note that this will not install the `gdcm` python package, because it is not in the PyPI package repository.
Having this package installed is important to be able to decompress some of the less common DICOM transfer
syntaxes (compression methods). If you are using Anaconda, you can install the `gdcm` package from their repositories.
On a Mac, you can install `gdcm` with:

```bash
brew install gdcm --with-python3
```

Alternatively, you can get the `gdcm` package by directly downloading a release from the
[github repository](https://github.com/malaterre/GDCM/releases) or installing from source (see the Dockerfile for an
example of this process). Note that you must have not just the C++ library, but also the Python bindings for Python 3
installed.
