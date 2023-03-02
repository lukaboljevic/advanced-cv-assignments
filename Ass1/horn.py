import numpy as np
from scipy.signal import convolve2d
from utils import gausssmooth, gaussderiv


def horn_schunck(img1, img2, num_iters, lmbd):
    """
    Estimate optical flow using Horn-Schunck algorithm

    Parameters:
        img1      - first image matrix (in grayscale; this is "frame t")
        img2      - second image matrix (in grayscale; this is "frame t+1")
        num_iters - number of iterations for improving optical flow
        lmbd      - parameter lambda, which serves as a regularization
                    constant, denoting how "strongly" do we want to value
                    smoothness of the energy field.

    Output:
        u, v - matrices containing the horizontal and vertical 
               (respectively) displacement components for each pixel
    """

    # Initialize
    u = np.zeros(img1.shape)
    v = np.zeros(img2.shape)
    sigma = 0.005 * min(img1.shape) # sigma for Gaussian filter is generally calculated like this
    avg_kernel = np.array([
        [0,   1/4, 0  ],
        [1/4, 0,   1/4],
        [0,   1/4, 0  ],
    ])

    
    # Calculate spatial and temporal derivatives
    It = gausssmooth(img2 - img1, sigma)
    img1deriv = gaussderiv(img1, sigma)  # TODO how much to set sigma here ...
    img2deriv = gaussderiv(img2, sigma)  # TODO same...
    Ix = 1/2 * (img1deriv[0] + img2deriv[0])
    Iy = 1/2 * (img1deriv[1] + img2deriv[1])
    print(">>> Derivatives done")


    # Iteratively refine optical flow
    for i in range(1, num_iters+1):
        avg_u = convolve2d(u, avg_kernel, mode="same")
        avg_v = convolve2d(v, avg_kernel, mode="same")
        P = np.divide(
            It + np.multiply(Ix, avg_u) + np.multiply(Iy, avg_v),
            np.square(Ix) + np.square(Iy) + lmbd
        )

        u = avg_u - np.multiply(Ix, P)
        v = avg_v - np.multiply(Iy, P)

        if i % 50 == 0:
            print(f"Iteration {i} done")

    print("Horn-Schunck done\n")
    return u, v