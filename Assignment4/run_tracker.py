from particle_tracker import ParticleTracker
from utils import RW, NCV, NCA

from assignment2.run_tracker import run_tracker


if __name__ == "__main__":
    # Path to dataset and sequence
    dataset_path = "../vot2014"
    sequence_name = "ball"

    # Parameters for particle tracker
    # dist_sigma should always be ~0.1 as it converts distance to probability the nicest:
    # https://www.desmos.com/calculator/eifdw6hevn
    particle_params = {
        "motion_model":     NCV,  # motion model to use
        "num_particles":    125,  # number of particles
        "q":                3,  # power spectral density of system covariance matrix Q
        "alpha":            0.01,  # update speed
        "sigma":            1,  # Sigma for Epanechnikov kernel
        "dist_sigma":       0.1,  # Sigma^2 for converting distance into probability
        "num_bins":         16,  # Number of bins when extracting histograms from patches
    }

    # Run the tracker
    run_tracker(ParticleTracker, particle_params, dataset_path, sequence_name)
