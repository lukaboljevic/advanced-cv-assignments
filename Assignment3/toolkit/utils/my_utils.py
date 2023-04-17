import math
import cv2
import numpy as np


# Particle tracker motion model constants
RW = 1
NCV = 2
NCA = 3


def create_cosine_window(target_size):
    """
    Create a cosine (Hanning) window. Parameter target_size is in the format 
    (width, height). The output is a cosine window of same size.
    """
    return cv2.createHanningWindow((target_size[0], target_size[1]), cv2.CV_32F)


def create_gauss_peak(target_size, sigma):
    """
    Create a Gaussian peak. Parameter target_size is in the format (width, height). 
    Sigma is the parameter (float) of the Gaussian function. 

    Note that sigma should be small so that the function is in a shape of a peak.
    Values that make sense are approximately from the interval: ~(0.5, 5)
    
    Output is a matrix of dimensions (width, height).
    """
    
    w2 = math.floor(target_size[0] / 2)
    h2 = math.floor(target_size[1] / 2)
    [X, Y] = np.meshgrid(np.arange(-w2, w2 + 1), np.arange(-h2, h2 + 1))
    G = np.exp(-X**2 / (2 * sigma**2) - Y**2 / (2 * sigma**2))
    G = np.roll(G, (-h2, -w2), (0, 1))
    return G


def get_patch(image, center, size):
    """
    Extract the patch from a given image, given center coordinates as (x, y) i.e.
    (column, row), and the size of the region to be extracted.

    Copied from Assignment 2.
    """
    # Crop coordinates
    x0 = round(int(center[0] - size[0] / 2))
    y0 = round(int(center[1] - size[1] / 2))
    x1 = int(round(x0 + size[0]))
    y1 = int(round(y0 + size[1]))

    # Padding
    x0_pad = max(0, -x0)
    x1_pad = max(x1 - image.shape[1] + 1, 0)
    y0_pad = max(0, -y0)
    y1_pad = max(y1 - image.shape[0] + 1, 0)

    # Crop target
    if len(image.shape) > 2:
        img_crop = image[y0 + y0_pad:y1 - y1_pad, x0 + x0_pad:x1 - x1_pad, :]
    else:
        img_crop = image[y0 + y0_pad:y1 - y1_pad, x0 + x0_pad:x1 - x1_pad]

    im_crop_padded = cv2.copyMakeBorder(img_crop, y0_pad, y1_pad, x0_pad, x1_pad, cv2.BORDER_REPLICATE)

    # Crop mask tells which pixels are within the image (1) and which are outside (0)
    m_ = np.ones((image.shape[0], image.shape[1]), dtype=np.float32)
    crop_mask = m_[y0 + y0_pad:y1 - y1_pad, x0 + x0_pad:x1 - x1_pad]
    crop_mask = cv2.copyMakeBorder(crop_mask, y0_pad, y1_pad, x0_pad, x1_pad, cv2.BORDER_CONSTANT, value=0)

    return im_crop_padded, crop_mask


def sample_gauss(mu, sigma, n):
    """
    Sample n samples from a multivariate normal distribution.

    Copied from Assignment 4.
    """
    return np.random.multivariate_normal(mu, sigma, n)


def epanechnikov_kernel(width, height, sigma):
    """
    Create Epanechnikov kernel of given size.

    Width and height need to be odd.

    Copied from Assignment 2.
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


def extract_and_normalize_hist(patch, nbins, weights=None):
    """
    Wrapper function that first calls `extract_histogram`, and then `normalize_histogram`.

    Copied from `extract_histogram` doc comment:
    "Extract a color histogram form a given patch, where the number of bins is the number of
    colors in the reduced color space.

    Note that the input patch must be a BGR image (3 channel numpy array). The function 
    thus returns a histogram nbins**3 bins."
    """
    histogram = extract_histogram(patch, nbins, weights)
    return normalize_histogram(histogram)


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

    Copied from Assignment 2.
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


if __name__ == "__main__":
    from screeninfo import get_monitors
    monitor = get_monitors()[0]
    w, h = monitor.width, monitor.height

    resize_to = 400  # resize Gaussian response to this size
    center_x = w // 2 - resize_to // 2
    center_y = h // 2 - resize_to // 2

    sigma = 5
    size = 15
    response = create_gauss_peak((size, size), sigma)
    response = cv2.resize(response, (resize_to, resize_to))

    window_name = f"Gaussian peak - size = {size}, sigma = {sigma}"
    cv2.imshow(window_name, response)
    cv2.moveWindow(window_name, center_x, center_y)
    cv2.waitKey(0)