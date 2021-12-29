#!/bin/bash
# A reminder on how to publish to the Python package index.
# - <https://packaging.python.org/tutorials/packaging-projects/>
# - <https://packaging.python.org/guides/distributing-packages-using-setuptools/>
#
# To work in development mode, install the library in development mode:
#   pip3 install --editable .

set -euo pipefail
cd "$(dirname $(realpath $0))"/..

# To build the package:
python3 -m pip install --upgrade build
python3 -m build

# To upload it:
python3 -m pip install --user --upgrade twine
twine upload dist/*
