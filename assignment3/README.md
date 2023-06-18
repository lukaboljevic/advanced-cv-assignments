# Correlation tracker

This assignment was about discriminative correlation tracking.

The implemented tracker was integrated in the VOT lite tracking toolkit, found [here](https://github.com/alanlukezic/pytracking-toolkit-lite).

The folder structure of this assignment is as follows. I edited some scripts from the lite toolkit, but the original functionality is completely preserved.

```
./
├── toolkit/                        # The lite version of the toolkit
│   ├── examples
│       ├── correlation_tracker.py  # Implementation of correlation tracker
│       └── particle_tracker.py     # Implementation of particle tracker (Assignment 4)
│
│   ├── utils
│       ├── ...                     # Various scripts which are part of the toolkit (I slightly edited some)
│       └── my_utils.py             # Utils required for the trackers, defined outside of the toolkit
│
│   ├── calculate_measures.py       # This, and the next 4 scripts were part of the lite toolkit.
│   ├── compare_trackers.py         # They were slightly edited so they can accept an additional argument: path to parameters of a tracker
│   ├── create_workspace.py
│   ├── evaluate_tracker.py
│   ├── visualize_result.py
│   ├── find_best.py                # Quick script to find the best correlation/particle tracker
│   └── test_params.py              # Functions to test the correlation and particle (Assignment 4) trackers
│
├── workspace-vot2014/
│   ├── analysis
│       ├── correlation*            # A bunch of results for the correlation tracker
│       └── particle*               # A bunch of results for the particle tracker
│   └── trackers.yaml               # Specification of implemented trackers as required by the toolkit
│
├── *.sh                            # Quick bash scripts to run the Python scripts from the toolkit to test/visualize tracker performance
└── *.json                          # JSON files specifying the hyperparameters for the correlation/particle tracker
```
