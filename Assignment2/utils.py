import math

import numpy as np
import cv2


def create_cosine_window(target_size):
    """
    Create a cosine (Hanning) window. Parameter target_size is in the format 
    (width, height). The output is a cosine window of same size.

    This function is used for correlation tracker only. Correlation tracker is
    part of Assignment 3.
    """
    return cv2.createHanningWindow((target_size[0], target_size[1]), cv2.CV_32F)


def create_gauss_peak(target_size, sigma):
    """
    Create a Gaussian peak. Parameter target_size is in the format (width, height). 
    Sigma is the parameter (float) of the Gaussian function. 

    Note that sigma should be small so that the function is in a shape of a peak.
    Values that make sense are approximately from the interval: ~(0.5, 5)
    
    Output is a matrix of dimensions (width, height).

    This function is used for correlation tracker only. Correlation tracker is
    part of Assignment 3.
    """
    w2 = math.floor(target_size[0] / 2)
    h2 = math.floor(target_size[1] / 2)
    [X, Y] = np.meshgrid(np.arange(-w2, w2 + 1), np.arange(-h2, h2 + 1))
    G = np.exp(-X**2 / (2 * sigma**2) - Y**2 / (2 * sigma**2))
    G = np.roll(G, (-h2, -w2), (0, 1))
    return G


def gausssmooth(img, sigma):
    """
    Perform Gauss smoothing on an image.

    Copied from Assignment 1.
    """
    x = np.array(list(range(math.floor(-3.0 * sigma + 0.5), math.floor(3.0 * sigma + 0.5) + 1)))
    G = np.exp(-x**2 / (2 * sigma**2))
    G = G / np.sum(G)
    return cv2.sepFilter2D(img, -1, G, G)


def generate_responses_1():
    """
    Generate artificial PDF.
    """
    responses = np.zeros((100, 100), dtype=np.float32)
    responses[70, 50] = 1
    responses[50, 70] = 0.5
    return gausssmooth(responses, 10)  # since the "PDF" needs to be smooth


def generate_responses_2():
    """
    Generate another artificial PDF.
    """
    responses = np.zeros((100, 100), dtype=np.float32)
    responses[35, 40] = 0.3
    responses[40, 70] = 1.2
    responses[65, 50] = 0.5
    return gausssmooth(responses, 10)


def get_patch(img, center, size):
    """
    Extract the patch from a given image, given center coordinates as (x, y) i.e.
    (column, row), and the size of the region to be extracted.
    """

    # Crop coordinates
    x0 = round(int(center[0] - size[0] / 2))
    y0 = round(int(center[1] - size[1] / 2))
    x1 = int(round(x0 + size[0]))
    y1 = int(round(y0 + size[1]))

    # Padding
    x0_pad = max(0, -x0)
    x1_pad = max(x1 - img.shape[1] + 1, 0)
    y0_pad = max(0, -y0)
    y1_pad = max(y1 - img.shape[0] + 1, 0)

    # Crop target
    if len(img.shape) > 2:
        img_crop = img[y0 + y0_pad:y1 - y1_pad, x0 + x0_pad:x1 - x1_pad, :]
    else:
        img_crop = img[y0 + y0_pad:y1 - y1_pad, x0 + x0_pad:x1 - x1_pad]

    im_crop_padded = cv2.copyMakeBorder(img_crop, y0_pad, y1_pad, x0_pad, x1_pad, cv2.BORDER_REPLICATE)

    # Crop mask tells which pixels are within the image (1) and which are outside (0)
    m_ = np.ones((img.shape[0], img.shape[1]), dtype=np.float32)
    crop_mask = m_[y0 + y0_pad:y1 - y1_pad, x0 + x0_pad:x1 - x1_pad]
    crop_mask = cv2.copyMakeBorder(crop_mask, y0_pad, y1_pad, x0_pad, x1_pad, cv2.BORDER_CONSTANT, value=0)

    return im_crop_padded, crop_mask


def coordinates_kernels(kernel_size):
    """
    Generate the coordinate kernels, labeled as x_i in the formula for mean shift
    """
    # Size needs to be odd
    if kernel_size[0] % 2 == 0 or kernel_size[1] % 2 == 0:
        raise ValueError(f"Kernel size not odd: {kernel_size}")
    
    value_x = kernel_size[0] // 2
    value_y = kernel_size[1] // 2

    # Option 1
    # arr_x = np.arange(-value_x, value_x + 1)
    # arr_y = np.arange(-value_y, value_y + 1).reshape((kernel_size[1], 1))
    # coordinates_x = np.tile(arr_x, (kernel_size[0], 1))  # copy array by rows
    # coordinates_y = np.tile(arr_y, (1, kernel_size[1]))  # copy by columns

    # Option 2
    # x = np.linspace(-value_x, value_x, kernel_size[0])
    # y = np.linspace(-value_y, value_y, kernel_size[1])
    # coordinates_x, coordinates_y = np.meshgrid(x, y)

    # Option 3
    x = np.arange(-value_x, value_x + 1)
    y = np.arange(-value_y, value_y + 1)
    coordinates_x, coordinates_y = np.meshgrid(x, y)

    return coordinates_x, coordinates_y


def epanechnikov_kernel(width, height, sigma):
    """
    Create Epanechnikov kernel of given size.

    Width and height need to be odd.
    """
    w2 = int(math.floor(width / 2))
    h2 = int(math.floor(height / 2))

    [X, Y] = np.meshgrid(np.arange(-w2, w2 + 1), np.arange(-h2, h2 + 1))
    X = X / np.max(X)
    Y = Y / np.max(Y)

    kernel = (1 - ((X / sigma)**2 + (Y / sigma)**2))
    kernel = kernel / np.max(kernel)
    kernel[kernel < 0] = 0
    return kernel


def gaussian_kernel(width, height, sigma="auto"):
    """
    Create Gaussian kernel of given size.

    Width and height need to be odd.
    """
    if sigma == "auto":
        # Formula is courtesy of OpenCV: https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gac05a120c1ae92a6060dd0db190a61afa
        sigma = 0.3 * ((min(width, height) - 1) * 0.5 - 1) + 0.8

    w2 = int(math.floor(width / 2))
    h2 = int(math.floor(height / 2))

    [X, Y] = np.meshgrid(np.arange(-w2, w2 + 1), np.arange(-h2, h2 + 1))
    X = X / np.max(X)
    Y = Y / np.max(Y)

    kernel = np.exp(-(np.square(X) + np.square(Y)) / (2 * sigma**2)) / (2 * np.pi * sigma**2)
    kernel = kernel / np.max(kernel)
    return kernel


def normalize_histogram(histogram):
    """
    Normalize a histogram, so that the sum of values is 1.
    """
    return histogram / np.sum(histogram)


def extract_histogram(patch, nbins, weights=None):
    """
    Extract a color histogram form a given patch, where the number of bins is the number of
    colors in the reduced color space.

    Note that the input patch must be a BGR image (3 channel numpy array). The function 
    thus returns a histogram nbins**3 bins.
    """

    # Convert each pixel intensity to the one of nbins bins
    channel_bin_idxs = np.floor((patch.astype(np.float32) / float(255)) * float(nbins - 1))
    # Calculate bin index of a 3D histogram
    bin_idxs = (channel_bin_idxs[:, :, 0] * nbins**2  + channel_bin_idxs[:, :, 1] * nbins + channel_bin_idxs[:, :, 2]).astype(np.int32)

    # Count bin indices to create histogram (use per-pixel weights if given)
    if weights is not None:
        histogram_ = np.bincount(bin_idxs.flatten(), weights=weights.flatten())
    else:
        histogram_ = np.bincount(bin_idxs.flatten())
    
    # Zero-pad histogram (needed since bincount function does not generate histogram with nbins**3 elements)
    histogram = np.zeros((nbins**3, 1), dtype=histogram_.dtype).flatten()
    histogram[:histogram_.size] = histogram_
    return histogram


def backproject_histogram(patch, histogram, nbins):
    """
    Backproject a histogram onto an extracted patch.
    
    Note that the input patch must be a BGR image (3 channel numpy array).
    """
    # Convert each pixel intensity to the one of nbins bins
    channel_bin_idxs = np.floor((patch.astype(np.float32) / float(255)) * float(nbins - 1))
    # Calculate bin index of a 3D histogram
    bin_idxs = (channel_bin_idxs[:, :, 0] * nbins**2  + channel_bin_idxs[:, :, 1] * nbins + channel_bin_idxs[:, :, 2]).astype(np.int32)

    # Use histogram us a lookup table for pixel backprojection
    backprojection = np.reshape(histogram[bin_idxs.flatten()], (patch.shape[0], patch.shape[1]))
    return backprojection


# base class for tracker
class Tracker():
    def __init__(self, params):
        self.parameters = params

    def initialize(self, image, region):
        raise NotImplementedError

    def track(self, image):
        raise NotImplementedError
