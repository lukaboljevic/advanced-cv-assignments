import argparse
import json
import os

from utils.utils import load_dataset, load_tracker
from utils.export_utils import export_plot, load_output, print_summary
from calculate_measures import tracking_analysis


def tracking_comparison(workspace_path, tracker_ids, sensitivity, output_path, params_path=None):

    dataset = load_dataset(workspace_path)

    outputs_all = []
    for tracker_id in tracker_ids:

        tracker_class = load_tracker(workspace_path, tracker_id)
        if params_path:
            with open(params_path, "r") as f:
                params = json.load(f)
            tracker = tracker_class(**params)
        else:
            tracker = tracker_class()

        results_path = os.path.join(workspace_path, 'analysis', tracker.name(), 'results.json')
        if os.path.exists(results_path):
            output = load_output(results_path)
            print_summary(output)
        else:
            output = tracking_analysis(workspace_path, tracker_id)
        
        outputs_all.append(output)

    if output_path == '':
        output_path = os.path.join(workspace_path, 'analysis', tracker.name(), 'ar.png')

    export_plot(outputs_all, sensitivity, output_path)


def main():
    parser = argparse.ArgumentParser(description='Tracking Visualization Utility')

    parser.add_argument('--workspace_path', help='Path to the VOT workspace', required=True, action='store')
    parser.add_argument('--trackers', help='Tracker identifiers', required=True, action='store', nargs='*')
    parser.add_argument('--params_path', help='Path to correlation filter params JSON file', required=False)
    parser.add_argument('--sensitivity', help='Sensitivtiy parameter for robustness', default=100, type=int)
    parser.add_argument('--output_path', help='Path for the output image', default='', type=str)

    args = parser.parse_args()
    
    tracking_comparison(args.workspace_path, args.trackers, args.sensitivity, args.output_path, args.params_path)

if __name__ == "__main__":
    main()