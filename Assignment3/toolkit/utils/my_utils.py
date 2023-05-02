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