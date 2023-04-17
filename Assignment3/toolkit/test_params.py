import random
import json
import os

from evaluate_tracker import evaluate_tracker
from calculate_measures import tracking_analysis
from utils.my_utils import RW, NCV, NCA


def test_correlation():
    workspace_path = "../workspace-vot2014"
    tracker = "correlation_tracker"
    params_path = "../correlation_tracker_params.json"

    sigmas = [0.5, 1, 2, 3, 4, 5]
    alphas = [0.02, 0.05, 0.1, 0.15, 0.2]
    enlarge_factors = [1, 1.5, 2, 2.5]
    lmbd = 0

    combinations = [(x, y, z) for x in sigmas for y in alphas for z in enlarge_factors]
    # num_tests = 19
    # tested = set()
    # i = 1

    # while i <= num_tests:
    for i in range(len(combinations)):
        # print(f"Testing tracker {i} / {num_tests}")
        print(f"Testing tracker {i+1} / {len(combinations)}")

        # sigma = random.choice(sigmas)
        # alpha = random.choice(alphas)
        # enlarge_factor = random.choice(enlarge_factors)
        # if (sigma, alpha, enlarge_factor) in tested:
        #     continue

        sigma, alpha, enlarge_factor = combinations[i]
        tracker_name = f"correlation-{enlarge_factor}ef-{sigma}sig-{alpha}al-{lmbd}lmb"
        print(f"Tracker name: {tracker_name}")

        analysis_path = f"{workspace_path}/analysis/{tracker_name}"
        if os.path.exists(analysis_path):
            continue

        if enlarge_factor == 1 and sigma == 3 and alpha == 0.02:
            # This one is so bad it breaks the damn code
            continue

        # tested.add((sigma, alpha, enlarge_factor))

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
        # i += 1


def test_particle():
    workspace_path = "../workspace-vot2014"
    tracker = "particle_tracker"
    params_path = "../particle_tracker_params.json"
    
    motion_model = NCV
    num_particles_list = [75, 100, 125, 150]
    qs = [1, 3, 5, 10, 15]
    alphas = [0, 0.001, 0.01, 0.05, 0.1]
    dist_sigma = 0.1
    num_bins_list = [16, 32]

    combinations = [(a, b, c, d) for a in num_particles_list
                                 for b in qs
                                 for c in alphas
                                 for d in num_bins_list]

    for i in range(len(combinations)):
        print(f"Testing tracker {i+1} / {len(combinations)}")

        num_particles, q, alpha, num_bins = combinations[i]
        mm = "RW" if motion_model == RW else "NCV" if motion_model == NCV else "NCA"
        tracker_name = f"particle-{mm}mm-{num_particles}N-{q}q-{alpha}al-{dist_sigma}dsig-{num_bins}nbins"
        print(f"Tracker name: {tracker_name}")

        analysis_path = f"{workspace_path}/analysis/{tracker_name}"
        if os.path.exists(analysis_path):
            continue

        params = {
            "motion_model": motion_model,
            "num_particles": num_particles,
            "q": q,
            "alpha": alpha,
            "sigma": 1,
            "dist_sigma": dist_sigma,
            "num_bins": num_bins
        }

        with open(params_path, "w") as f:
            json.dump(params, f, indent=4)
        
        evaluate_tracker(workspace_path, tracker, params_path)
        tracking_analysis(workspace_path, tracker, params_path)
        print()
        print()


if __name__ == "__main__":
    # test_correlation()
    test_particle()