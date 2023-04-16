#!/bin/bash

cd toolkit
python visualize_result.py --workspace_path ../workspace-vot2014 --tracker particle_tracker \
--params_path ../particle_tracker_params.json --sequence $1 --show_gt