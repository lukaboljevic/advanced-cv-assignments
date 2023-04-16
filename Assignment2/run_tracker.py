import time

import cv2
import numpy as np

from sequence_utils import VOTSequence
from ms_tracker import MSTracker
from correlation_tracker import CorrelationTracker  # Part of Assignment 3, not Assignment 2


def run_tracker(tracker_class, tracker_params, dataset_path, sequence_name):
    """
    Run and print results of tracker.

    Parameters
    ----------
    tracker_class : Tracker class (eg. MSTracker, CorrelationTracker)
    tracker_params : Parameters for the tracker as a dictionary
    dataset_path : Path to the dataset with sequences
    sequence_name :  Dataset sequence to test the tracker on
    """

    """
    Note:
    Sequence "vot2014/ball" has a bug, where the number of ground truths is 602, but the number of frames
    is 603. You can fix that by removing the very first frame, or very last frame. Other sequences are fine.
    """
    # Initialize tracker object
    tracker = tracker_class(**tracker_params)


    # Visualization and setup parameters
    win_name = f"{tracker.__class__.__name__} - {sequence_name}"
    reinitialize = True
    show_gt = True
    video_delay = 15
    font = cv2.FONT_HERSHEY_PLAIN


    # Create sequence object
    sequence = VOTSequence(dataset_path, sequence_name)
    init_frame = 0
    n_failures = 0


    # Initialize visualization window
    sequence.initialize_window(win_name)


    # Tracking loop - goes over all frames in the video sequence
    frame_idx = 0
    time_all = 0
    overlap_values = []
    while frame_idx < sequence.length():
        img = cv2.imread(sequence.frame(frame_idx))

        # Initialize or track
        if frame_idx == init_frame:
            # Initialize tracker (at the beginning of the sequence or after tracking failure)
            t_ = time.time()
            tracker.initialize(img, sequence.get_annotation(frame_idx, type="rectangle"))
            time_all += time.time() - t_
            predicted_bbox = sequence.get_annotation(frame_idx, type="rectangle")
        else:
            # Track on current frame - predict bounding box
            t_ = time.time()
            predicted_bbox = tracker.track(img)
            time_all += time.time() - t_

        # Calculate overlap (needed to determine failure of a tracker)
        gt_bb = sequence.get_annotation(frame_idx, type="rectangle")
        overlap = sequence.overlap(predicted_bbox, gt_bb)
        overlap_values.append(overlap)

        # Draw ground-truth and predicted bounding boxes, frame numbers and show image
        if show_gt:
            sequence.draw_region(img, gt_bb, (0, 255, 0), 1)
        sequence.draw_region(img, predicted_bbox, (0, 0, 255), 2)
        sequence.draw_text(img, "%d/%d" % (frame_idx + 1, sequence.length()), (15, 25))
        sequence.draw_text(img, "Fails: %d" % n_failures, (15, 55))
        sequence.show_image(img, video_delay)

        if overlap > 0 or not reinitialize:  # I'm not really sure if overlap > 0 is appropriate, but just leave it
        # if overlap > 0.1 or not reinitialize:
            # Increase frame counter by 1
            frame_idx += 1
        else:
            # Increase frame counter by 5 and set re-initialization to the next frame
            frame_idx += 5
            init_frame = frame_idx
            n_failures += 1


    # Print results
    avg_overlap = sum(overlap_values) / sequence.length()
    median_overlap = np.median(overlap_values)
    print(f"Sequence: {dataset_path}/{sequence_name}")
    print("Tracking speed: %.1f FPS" % (sequence.length() / time_all))
    print("Tracker failed %d times" % n_failures)
    print(f"Mean overlap: {(avg_overlap):.4f}")
    print(f"Median overlap: {(median_overlap):.4f}")


if __name__ == "__main__":
    # Path to dataset and sequence
    dataset_path = "vot2014"
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