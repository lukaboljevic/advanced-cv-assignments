from correlation_tracker import CorrelationTracker  # Part of Assignment 3, not Assignment 2
from ms_tracker import MSTracker
from run_tracker import run_tracker


if __name__ == "__main__":
    # Path to dataset and sequence
    dataset_path = "../vot2014"
    sequence_name = "ball"

    # Parameters for MS tracker
    ms_params = {
        "kernel_type":  "epanechnikov",
        # "kernel_type":  "gaussian",
        "sigma":        1,  # for Epanechnikov or Gaussian kernel; for Gaussian, you can also specify 
                            # "auto", and sigma will be calculated based on the formula 
                            # 0.3*((min(kernel_shape)-1)*0.5 - 1) + 0.8 (courtesy of OpenCV)
        "num_bins":     16,  # number of bins used for the target template/candidate histograms
        "alpha":        0.0,  # update speed (how much do we update self.q after localizing the target in next frame)
        "eps_stop":     1,  # stop mean shift iterations when changes in x and y drop below this value
        "eps_v":        1e-3,  # used for numerical stability when calculating weights `v` during mean shift
    }

    # Parameters for correlation tracker (part of Assignment 3)
    correlation_params = {
        "sigma":            2,  # sigma for ideal Gaussian response
        "lmbd":             0,  # controls how large values in filter H are (check cost formula function)
        "alpha":            0.1,  # update speed for filter H
        "enlarge_factor":   1.5,  # by how much our target search region is enlarged
    }

    # Run the tracker
    run_tracker(MSTracker, ms_params, dataset_path, sequence_name)
    
    # Correlation tracker is part of Assignment 3
    # run_tracker(CorrelationTracker, correlation_params, dataset_path, sequence_name)