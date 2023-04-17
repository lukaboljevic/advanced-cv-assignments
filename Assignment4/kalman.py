import math
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from utils import kalman_step, RW, NCV, NCA


def define_matrices(motion_model, q, r):
    """
    Define the necessary matrices for Kalman filter, based on the motion model.

    Parameters
    ----------
    motion_model : which motion model to use, can be one of the following constants:
        - RW (random walk), 
        - NCV (nearly constant velocity)
        - NCA (nearly constant acceleration)
    q : spectral density of the system covariance matrix Q
    r : spectral density of the observation covariance matrix R. 
    
    Returns
    -------
    Φ (A in `kalman_step`) : system matrix
    H (C in `kalman_step`) : observation matrix
    Q (Q in `kalman_step`) : system covariance matrix
    R (R in `kalman_step`) : observation covariance matrix
    """

    T = sp.symbols("T")
    
    # For all motion models, R = r*[[1, 0], [0, 1]], based on project slides
    R = r * np.array([[1, 0], [0, 1]], dtype=np.float64)

    # Matrices Fi, H and Q depend on the motion model. Fi and Q are calculated in the 
    # same way for all motion models, but they depend on the state transition matrix F,
    # and on Fi and another matrix L, respectively. H just depends on the motion model.
    if motion_model == RW:
        # State is defined as X = [x, y]^T
        F = sp.Matrix([
            [0, 0],
            [0, 0]
        ])
        L = sp.Matrix([
            [1, 0],
            [0, 1]
        ])
        H = np.array([
            [1, 0],
            [0, 1]
        ], dtype=np.float64)

    elif motion_model == NCV:
        # State is defined as X = [x, x', y, y']^T
        # x' and y' denote the velocities in x and y directions
        F = sp.Matrix([
            [0, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ])
        L = sp.Matrix([
            [0, 0],
            [1, 0],
            [0, 0],
            [0, 1]
        ])
        H = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0]
        ], dtype=np.float64)

    elif motion_model == NCA:
        # State is defined as X = [x, x', x'', y, y', y'']^T
        # x' and y' denote the velocities in x and y directions
        # x'' and y'' denote the accelerations in x and y directions
        F = sp.Matrix([
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0]
        ])
        L = sp.Matrix([
            [0, 0],
            [0, 0],
            [1, 0],
            [0, 0],
            [0, 0],
            [0, 1]
        ])
        H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0]
        ], dtype=np.float64)

    # Calculate Fi, based on lecture slides
    Fi = sp.exp(F * T)

    # Calculate Q, based on lecture slides
    temp = Fi * L
    Q = sp.integrate(temp * q * temp.T, (T, 0, T))  # integrate w.r.t. T, from 0 to T

    # Replace T with 1 (1 since we're processing "frame-by-frame") and convert to numpy
    Fi = np.array(Fi.replace(T, 1).tolist(), dtype=np.float64)
    Q = np.array(Q.replace(T, 1).tolist(), dtype=np.float64)

    return Fi, H, Q, R


def spiral_trajectory(N):
    """
    Create an artificial spiral trajectory.
    """
    v = np.linspace(5 * math.pi, 0, N)
    x = np.cos(v) * v
    y = np.sin(v) * v
    
    return x, y


def main(motion_model, q, r, N):
    # Artificial spiral trajectory
    x, y = spiral_trajectory(N)

    plt.figure(figsize=(8, 7))
    plt.plot(x, y, color="firebrick", label="Original")
    plt.scatter(x, y, marker="o", facecolors="none", edgecolors="firebrick", s=30)

    # Matrices for Kalman filter
    Fi, H, Q, R = define_matrices(motion_model, q, r)

    # Keep track of Kalman filter observations/measurements
    observations_x = np.zeros((x.size, 1), dtype=np.float32).flatten()
    observations_y = np.zeros((y.size, 1), dtype=np.float32).flatten()
    observations_x[0] = x[0]
    observations_y[0] = y[0]

    # Initialize state and prior covariance
    """
    Currently:
        RW state:  [x, y]^T
        NCV state: [x, x', y, y']^T
        NCA state: [x, x', x'', y, y', y'']^T

    State is initialized based on motion model - notice that only the placement of y coordinate
    is different - for RW state[1], for NCV state[2], for NCA state[3]. This is because of the
    way we defined F, L and H matrices in define_matrices. If we want the state to always have
    x, y as the first two components, F, L and H matrices would have to be slightly changed.
    """
    state = np.zeros((Fi.shape[0], 1), dtype=np.float32).flatten()
    state[0] = x[0]
    state[motion_model] = y[0]
    covariance = np.eye(Fi.shape[0], dtype=np.float32)

    # Perform Kalman steps
    for j in range(1, x.size):
        state, covariance, _, _ = kalman_step(Fi, H, Q, R,
                                              np.reshape(np.array([x[j], y[j]]), (-1, 1)),
                                              np.reshape(state, (-1, 1)),
                                              covariance)
        
        observations_x[j] = state[0]
        observations_y[j] = state[motion_model]

    # Plot
    plt.plot(observations_x, observations_y, color="limegreen", label="Kalman")
    plt.scatter(observations_x, observations_y, marker="o", facecolors="none", edgecolors="limegreen", s=30)

    mm = "RW" if motion_model == RW else "NCV" if motion_model == NCV else "NCA"
    plt.title(f"{mm}, q = {q}, r = {r}")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    motion_model = NCV
    q = 1
    r = 1
    N = 40

    main(motion_model, q, r, N)