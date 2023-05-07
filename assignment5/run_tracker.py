import argparse
import os
import cv2

from tools.sequence_utils import VOTSequence
from tools.sequence_utils import save_results
from siamfc import TrackerSiamFC


def evaluate_tracker(dataset_path, network_path, results_dir, visualize, long_term=False):
    sequences = []
    with open(os.path.join(dataset_path, 'list.txt'), 'r') as f:
        for line in f.readlines():
            sequences.append(line.strip())

    long_term_params = None
    if long_term:
        long_term_params = {
            "num_locations": 10,
            "scale_up": 1.0,
            "failure_threshold": 4.5 # if maximum of correlation response falls below the threshold, redetection starts
        }

    tracker = TrackerSiamFC(net_path=network_path, long_term=long_term, long_term_params=long_term_params)

    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    for sequence_name in sequences:
        print('Processing sequence:', sequence_name)

        bboxes_path = os.path.join(results_dir, '%s_bboxes.txt' % sequence_name)
        scores_path = os.path.join(results_dir, '%s_scores.txt' % sequence_name)
        if os.path.exists(bboxes_path) and os.path.exists(scores_path):
            print('Results on this sequence already exists. Skipping.')
            continue
        
        sequence = VOTSequence(dataset_path, sequence_name)
        img = cv2.imread(sequence.frame(0))
        gt_rect = sequence.get_annotation(0)
        tracker.init(img, gt_rect)
        results = [gt_rect]
        scores = [[10000]]  # a very large number - very confident at initialization

        if visualize:
            cv2.namedWindow('SiamFC window', cv2.WINDOW_AUTOSIZE)

        for i in range(1, sequence.length()):
            img = cv2.imread(sequence.frame(i))
            prediction, score, redetection_bboxes = tracker.update(img)
            results.append(prediction)
            scores.append([score])

            if visualize:
                sequence.draw_text(img, '%d/%d' % (i + 1, sequence.length()), (25, 25))
                sequence.draw_text(img, 'Score: %.3f' % score, (25, 50))

                if redetection_bboxes is None:
                    tl_ = (int(round(prediction[0])), int(round(prediction[1])))
                    br_ = (int(round(prediction[0] + prediction[2])), int(round(prediction[1] + prediction[3])))
                    cv2.rectangle(img, tl_, br_, (0, 0, 255), 2)
                else:
                    for bbox in redetection_bboxes:
                        tl_ = (int(round(bbox[0])), int(round(bbox[1])))
                        br_ = (int(round(bbox[0] + bbox[2])), int(round(bbox[1] + bbox[3])))
                        cv2.rectangle(img, tl_, br_, (20, 145, 230), 2)

                cv2.imshow('SiamFC window', img)
                key_ = cv2.waitKey(10)
                if key_ == 27:
                    exit(0)
        
        save_results(results, bboxes_path)
        save_results(scores, scores_path)

    if long_term_params is None:
        return
    
    with open(os.path.join(results_dir, "params.txt"), "w") as f:
        f.write(f"{long_term_params['num_locations']} locations\n")
        f.write(f"search size {long_term_params['scale_up']:.1f} * x_sz\n")
        f.write(f"{long_term_params['failure_threshold']} failure threshold\n")
        f.write("uniform sampling over entire image")


parser = argparse.ArgumentParser(description='SiamFC Runner Script')

parser.add_argument("--dataset", help="Path to the dataset", required=True, action='store')
parser.add_argument("--net", help="Path to the pre-trained network", required=True, action='store')
parser.add_argument("--results_dir", help="Path to the directory to store the results", required=True, action='store')
parser.add_argument("--visualize", help="Show ground-truth annotations", required=False, action='store_true')
parser.add_argument("--long_term", help="Whether to make SiamFC a long term tracker", required=False, action='store_true')

args = parser.parse_args()

evaluate_tracker(args.dataset, args.net, args.results_dir, args.visualize, args.long_term)
