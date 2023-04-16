#!/bin/bash

cd toolkit
python compare_trackers.py --workspace_path ../workspace-vot2014 --trackers correlation_tracker \
--params_path ../correlation_tracker_params.json --sensitivity 100