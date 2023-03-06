import numpy as np
from scipy.signal import convolve2d
from utils import gausssmooth, gaussderiv
from lucas import lucas_kanade


def horn_schunck(img1, img2, max_iters, lmbd, N=None, eps=None):
    """
    Estimate optical flow using Horn-Schunck algorithm

    Parameters:
        img1      - first image matrix (in grayscale; this is "frame t")

        img2      - second image matrix (in grayscale; this is "frame t+1")

        max_iters - number of iterations for improving optical flow
        
        lmbd      - parameter lambda, which serves as a regularization
                    constant, denoting how "strongly" do we want to value
                    smoothness of the energy field.

        N         - neighborhood size for Lucas-Kanade, given as a parameter
                    when we want to initialize u and v with the output from
                    Lucas-Kanade

        eps       - optional parameter, used for determining whether u and v
                    have "converged"

    Output:
        u, v - matrices containing the horizontal and vertical 
               (respectively) displacement components for each pixel
    """

    # Initialize necessary stuff
    if N:
        # Initialize with output of Lucas-Kanade
        u, v = lucas_kanade(img1, img2, N)
    else:
        u = np.zeros(img1.shape)
        v = np.zeros(img2.shape)
    avg_kernel = np.array([
        [0,   1/4, 0  ],
        [1/4, 0,   1/4],
        [0,   1/4, 0  ],
    ])
    sigma = 0.005 * min(img1.shape) # sigma for Gaussian filter is generally calculated like this

    
    # Calculate spatial and temporal derivatives
    It = gausssmooth(img2 - img1, sigma)
    img1deriv = gaussderiv(img1, sigma)
    img2deriv = gaussderiv(img2, sigma)
    Ix = 1/2 * (img1deriv[0] + img2deriv[0])
    Iy = 1/2 * (img1deriv[1] + img2deriv[1])
    P_bottom = np.square(Ix) + np.square(Iy) + lmbd  # so we don't recalculate unnecessarily
    print(">>> Derivatives done")


    # Iteratively refine optical flow until 'convergence' or until iteration limit reached
    for i in range(1, max_iters+1):
        avg_u = convolve2d(u, avg_kernel, mode="same")
        avg_v = convolve2d(v, avg_kernel, mode="same")
        P = np.divide(
            It + np.multiply(Ix, avg_u) + np.multiply(Iy, avg_v),
            P_bottom
        )

        prev_u = u
        prev_v = v
        u = avg_u - np.multiply(Ix, P)
        v = avg_v - np.multiply(Iy, P)

        u_mean_diff = np.mean(np.abs(u - prev_u))
        v_mean_diff = np.mean(np.abs(v - prev_v))

        if i % 50 == 0:
            print(f"\tIteration {i} done")
            print(f"\t\tu_mean diff: {u_mean_diff}")
            print(f"\t\tv_mean diff: {v_mean_diff}")

        if eps and u_mean_diff < eps and v_mean_diff < eps:
            print(f"\tConverged on iteration {i}")
            break

    print("Horn-Schunck done\n")
    return u, v