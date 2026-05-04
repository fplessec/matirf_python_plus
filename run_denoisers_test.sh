#!/bin/bash
cd "$(dirname "$0")"
source env/bin/activate
python -m algorithms.denoisers.tests.main
