import os
import json


def show_top(array, n, type):
    print(f"####### Sorted by {type} #######\n")
    for i in range(n):
        curr = array[i]
        print(f"Rank {i+1}")
        print(f"   Name: {curr['tracker_name']}")
        print(f"   Failures: {curr['total_failures']}")
        print(f"   Overlap: {curr['average_overlap']}")
        print(f"   Average speed: {(curr['average_speed']):.2f} FPS")
        print(f"   Average init speed: {(curr['average_init_speed']):.2f} FPS")
        print()
    print("--------------------------------------------------")
    print("--------------------------------------------------")


if __name__ == "__main__":
    root_path = "../workspace-vot2014/analysis"
    all_trackers = []

    for tracker_dir in os.listdir(root_path):
        results_json_path = os.path.join(root_path, tracker_dir, "results.json")
        with open(results_json_path, "r") as f:
            data = json.load(f)

        # curr_tracker_name: str = data["tracker_name"]
        # if curr_tracker_name.find("particle") < 0:
        #     continue

        all_trackers.append(data)

    by_failures = sorted(all_trackers, key=lambda tracker: tracker["total_failures"])
    by_overlap = sorted(all_trackers, key=lambda tracker: tracker["average_overlap"], reverse=True)
    by_speed = sorted(all_trackers, key=lambda tracker: tracker["average_speed"], reverse=True)

    n = 5

    show_top(by_failures, n, "failures")
    show_top(by_overlap, n, "overlap")
    show_top(by_speed, n, "speed")