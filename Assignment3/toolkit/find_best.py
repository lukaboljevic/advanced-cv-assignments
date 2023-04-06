import os
import json

root_path = "../workspace-vot2014/analysis"

highest_overlap = 0
highest_overlap_failures = 0
highest_overlap_name = ""
least_failures = 1000000
least_failures_overlap = 0
least_failures_name = ""

for tracker_dir in os.listdir(root_path):
    results_json_path = os.path.join(root_path, tracker_dir, "results.json")
    with open(results_json_path, "r") as f:
        data = json.load(f)

    curr_tracker_name = data["tracker_name"]
    curr_overlap = data["average_overlap"]
    curr_failures = data["total_failures"]

    if curr_overlap > highest_overlap:
        highest_overlap = curr_overlap
        highest_overlap_name = curr_tracker_name
        highest_overlap_failures = curr_failures

    if curr_failures < least_failures:
        least_failures = curr_failures
        least_failures_name = curr_tracker_name
        least_failures_overlap = curr_overlap


print("Best by total number of failures:")
print(f"\tName: {least_failures_name}\n\tTotal failures: {least_failures}\n\tAverage overlap: {least_failures_overlap}")
print()
print("Best by average overlap:")
print(f"\tName: {highest_overlap_name}\n\tAverage overlap: {highest_overlap}\n\tTotal failures: {highest_overlap_failures}")

"""
Best by total number of failures:
        Name: correlation-1.5ef-2sig-0.15al-0lmb
        Total failures: 64
        Average overlap: 0.4687953692471228     

Best by average overlap:
        Name: correlation-2ef-1sig-0.05al-0lmb  
        Average overlap: 0.49638076974183465    
        Total failures: 93
"""