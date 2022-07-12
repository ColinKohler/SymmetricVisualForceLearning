#!/bin/bash

pkill -9 ray
pipenv run python scripts/train.py peg_insertion --results_path=sm_tol_1
pkill -9 ray
pipenv run python scripts/train.py peg_insertion --results_path=sm_tol_2
pkill -9 ray
pipenv run python scripts/train.py peg_insertion --results_path=sm_tol_3
