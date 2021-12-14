#!/bin/bash
# Convenience script to run all tests.

cd "$(dirname $(realpath $0))"/..
python3 -m unittest tests/test_*.py
