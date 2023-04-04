import math
import cv2
import numpy as np


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