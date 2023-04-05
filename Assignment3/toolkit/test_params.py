import random
import json

from evaluate_tracker import evaluate_tracker
from calculate_measures import tracking_analysis


workspace_path = "../workspace-vot2014"
tracker = "correlation_tracker"
params_path = "../correlation_tracker_params.json"

sigmas = [1, 2, 3, 4]
alphas = [0.02, 0.05, 0.1, 0.15, 0.2]
enlarge_factors = [1, 1.5, 2, 2.5]
lmbd = 0

num_tests = 20
tested = set()
i = 1

while i <= num_tests:
    sigma = random.choice(sigmas)
    alpha = random.choice(alphas)
    enlarge_factor = random.choice(enlarge_factors)
    if (sigma, alpha, enlarge_factor) in tested:
        continue

    tested.add((sigma, alpha, enlarge_factor))

    params = {
        "sigma": sigma,
        "lmbd": lmbd,
        "alpha": alpha,
        "enlarge_factor": enlarge_factor
    }

    with open(params_path, "w") as f:
        json.dump(params, f, indent=4)
    
    evaluate_tracker(workspace_path, tracker, params_path)
    tracking_analysis(workspace_path, tracker, params_path)
    print()
    print()
    i += 1