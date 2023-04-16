#!/bin/bash

cd toolkit
python evaluate_tracker.py --workspace_path ../workspace-vot2014 --tracker particle_tracker \
--params_path ../particle_tracker_params.json