#!/bin/bash
set -e;
export PYTHONIOENCODING=utf8
python3 -m pytest -m 'not integration_test' -m 'not stress_test' --capture=no "$@" -vv
