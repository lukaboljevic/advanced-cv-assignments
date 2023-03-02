import numpy as np
from scipy.signal import convolve2d
from utils import gausssmooth, gaussderiv


def lucas_kanade(img1, img2, N=3):
    """
    Estimate optical flow using Lucas-Kanade algorithm

    Parameters:
        img1 - first image matrix (in grayscale; this is "frame t")
        img2 - second image matrix (in grayscale; this is "frame t+1")
        N    - size of the neighborhood considered, when calculating 
               displacement vectors for given pixel

    Output:
        u, v - matrices containing the horizontal and vertical 
               (respectively) displacement components for each pixel
    """

    # Initialize
    sum_kernel = np.ones((N, N))
    sigma = 0.005 * min(img1.shape)  # sigma for Gaussian filter is generally calculated like this


    # Calculate spatial and temporal derivatives
    It = gausssmooth(img2 - img1, sigma)
    img1deriv = gaussderiv(img1, sigma)  # TODO how much to set sigma here ...
    img2deriv = gaussderiv(img2, sigma)  # TODO same...
    Ix = 1/2 * (img1deriv[0] + img2deriv[0])
    Iy = 1/2 * (img1deriv[1] + img2deriv[1])
    print(">>> Derivatives done")


    # Calculate sum of required element-wise products
    sIxt = convolve2d(np.multiply(Ix, It), sum_kernel)
    sIyt = convolve2d(np.multiply(Iy, It), sum_kernel)
    sIxy = convolve2d(np.multiply(Ix, Iy), sum_kernel)
    sIxx = convolve2d(np.multiply(Ix, Ix), sum_kernel)
    sIyy = convolve2d(np.multiply(Iy, Iy), sum_kernel)
    print(">>> Convolutions done")


    # Calculate determinant of covariance matrix
    D = np.multiply(sIxx, sIyy) - np.square(sIxy) + 0.00001  # just in case D = 0


    # Calculate matrices containing displacement components - u, v
    u_top = np.multiply(sIxy, sIyt) - np.multiply(sIyy, sIxt)
    u = np.divide(u_top, D)

    v_top = np.multiply(sIxy, sIxt) - np.multiply(sIxx, sIyt)
    v = np.divide(v_top, D)
    print(">>> Lucas-Kanade done\n")

    return u, v