import numpy as np
from scipy.signal import convolve2d
from utils import gaussderiv, gausssmooth


def lucas_kanade(img1, img2, N=3, verbose=True):
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
    img1deriv = gaussderiv(img1, sigma)
    img2deriv = gaussderiv(img2, sigma)
    Ix = 1/2 * (img1deriv[0] + img2deriv[0])
    Iy = 1/2 * (img1deriv[1] + img2deriv[1])
    if verbose:
        print(">>> Derivatives done")


    # Calculate sum of required element-wise products
    sIxt = convolve2d(np.multiply(Ix, It), sum_kernel, mode="same")  # has to be same because we may use it for HS
    sIyt = convolve2d(np.multiply(Iy, It), sum_kernel, mode="same")
    sIxy = convolve2d(np.multiply(Ix, Iy), sum_kernel, mode="same")
    sIxx = convolve2d(np.multiply(Ix, Ix), sum_kernel, mode="same")
    sIyy = convolve2d(np.multiply(Iy, Iy), sum_kernel, mode="same")
    if verbose:
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