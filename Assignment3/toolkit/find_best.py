import os
import json

root_path = "../workspace-vot2014/analysis"

all_trackers = []

for tracker_dir in os.listdir(root_path):
    results_json_path = os.path.join(root_path, tracker_dir, "results.json")
    with open(results_json_path, "r") as f:
        data = json.load(f)

    # curr_tracker_name = data["tracker_name"]
    # enlarge_factor = float(curr_tracker_name.split("-")[1][:-2])
    # if enlarge_factor != 1.0:
    #     continue

    all_trackers.append(data)

by_failures = sorted(all_trackers, key=lambda tracker: tracker["total_failures"])
by_overlap = sorted(all_trackers, key=lambda tracker: tracker["average_overlap"], reverse=True)

n = 3

for i in range(n):
    curr = by_failures[i]
    print(f"Rank {i+1}")
    print(f"   Name: {curr['tracker_name']}")
    print(f"   Failures: {curr['total_failures']}")
    print(f"   Overlap: {curr['average_overlap']}")
    print(f"   Average speed: {(curr['average_speed']):.2f} FPS")
    print(f"   Average init speed: {(curr['average_init_speed']):.2f} FPS")
    print()

print("--------------------------------------------------")
print("--------------------------------------------------\n")

for i in range(n):
    curr = by_overlap[i]
    print(f"Rank {i+1}")
    print(f"   Name: {curr['tracker_name']}")
    print(f"   Failures: {curr['total_failures']}")
    print(f"   Overlap: {curr['average_overlap']}")
    print(f"   Average speed: {(curr['average_speed']):.2f} FPS")
    print(f"   Average init speed: {(curr['average_init_speed']):.2f} FPS")