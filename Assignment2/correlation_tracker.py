# ------------------------------------------------------------------------------------
# Note: This tracker is for Assignment 3, it is not part of Assignment 2. It was added
# here because it's useful and easier to test outside of Assignment 3 toolkit.
# ------------------------------------------------------------------------------------

import numpy as np
from numpy.fft import fft2, ifft2
import cv2

import utils as ut

"""
All extracted patches should be grayscaled, and first multiplied with a cosine (Hanning) window
before transforming to Fourier domain.

We are in frame 1, and we are given the patch where the target is. We first make the patch
grayscale, and pointwise multiply it with a Hanning window, to reduce boundary effects when
calculating FFT. We also create the ideal correlation response G (Gaussian peak). Then, we
calculate the filter H, from the target patch and response G. We use the closed form solution
formula for H to do that.

Now that we have the filter, we can track the target. To search for the target in frame 2,
we extract the patch where the target previously was. After multiplying with the cosine window,
we compute the correlation response G' from the extracted patch and current filter H. New target 
position is at the position of maximum peak in G'.

All we need to do now is update the filter H. We first calculate a new filter H', obtained
from the patch extracted at new target location. Don't forget to multiply with cosine window. 
With update speed alpha, we update filter H using formula 
    H = (1-alpha)*H + alpha*H'.


Implementation details:
    - As mentioned, all image patches, used for computing new filter(s) and for localization, 
    should be multiplied by a cosine (Hanning) window. Multiplication should be done before
    transforming to Fourier domain.
    - Always be sure to use grayscale patches
    - Ideal Gaussian response G should be calculated only once, during initialization

    - The size of the cosine window, Gaussian response, filter H and extracted image patches are 
    all the same. However, the tracker doesn't really perform well when the target is moving a lot. 
    One solution we can try is increasing the search range - extracting larger image patches and
    searching for the target there. We just need to pay attention that the predicted bounding box 
    still should have the same size as ground-truth bounding box.
"""


class CorrelationTracker(ut.Tracker):
    def __init__(self, **params):
        self.sigma = params.get("sigma", 2)  # sigma for ideal Gaussian response
        self.lmbd = params.get("lmbd", 0)  # controls how large values in filter H are (check cost formula function)
        self.alpha = params.get("alpha", 0.1)  # update speed for filter H
        self.enlarge_factor = params.get("enlarge_factor", 1.5)  # by how much our target search region is enlarged
        if self.enlarge_factor < 1:
            raise ValueError("Enlarge factor can't be smaller than 1.")


    def compute_H(self, F):
        """
        Return filter FFT(H).conjugate() calculated from ideal Gaussian response
        G, and target patch F, which has previously been grayscaled and multiplied
        with a cosine (Hanning) window
        """
        target_patch_fft = fft2(F)
        
        # Result is FFT(H).conjugate()
        H = np.divide(
            np.multiply(self.G_fft, target_patch_fft.conjugate()),
            np.multiply(target_patch_fft, target_patch_fft.conjugate()) + self.lmbd
        )

        # We don't really need to do any inverse FFT
        return H
    

    def initialize(self, image, region):
        # Param region - tuple of length 4; region[0] and region[1] are the x, y coordinates 
        # (respectively) of the top left corner of the ground truth bbox. region[2] and region[3] 
        # are the width and height (respectively) of the ground truth bbox

        # Convert to grayscale as we don't need colors
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if len(region) == 8:
            x_ = np.array(region[::2])
            y_ = np.array(region[1::2])
            region = [np.min(x_), np.min(y_), np.max(x_) - np.min(x_) + 1, np.max(y_) - np.min(y_) + 1]
        
        # Make sure that the width and height of the ground truth box are odd(!) integers
        region[2], region[3] = round(region[2]), round(region[3])
        if region[2] % 2 == 0:
            region[2] -= 1
        if region[3] % 2 == 0:
            region[3] -= 1


        # Target bounding box position and size
        self.position = (region[0] + region[2] / 2, region[1] + region[3] / 2)  # x, y coordinates of center of the target
        self.bbox_size = (region[2], region[3])  # width, height of the target bounding box

        # We will search for the target in a larger region
        enlarged_width = round(region[2] * self.enlarge_factor)
        enlarged_height = round(region[3] * self.enlarge_factor)
        if enlarged_width % 2 == 0:
            enlarged_width -= 1
        if enlarged_height % 2 == 0:
            enlarged_height -= 1
        self.search_size = (enlarged_width, enlarged_height)
        # enlarged_size = round(max(region[2], region[3]) * self.enlarge_factor)
        # if enlarged_size % 2 == 0:
        #     enlarged_size -= 1
        # self.search_size = (enlarged_size, enlarged_size)

        # Initial target template i.e. patch
        self.target_patch, _ = ut.get_patch(image, self.position, self.search_size)


        # Cosine (Hanning window) and ideal Gaussian response
        self.cosine_window = ut.create_cosine_window(self.search_size)
        self.G = ut.create_gauss_peak(self.search_size, self.sigma)
        self.G_fft = fft2(self.G)

        # Multiply template with cosine window to reduce the impact of boundary effects when doing FFT
        self.target_patch = np.multiply(self.target_patch, self.cosine_window)

        # Calculate initial filter H
        self.H = self.compute_H(self.target_patch)


    def track(self, image):
        # Convert to grayscale as we don't need colors
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


        # Extract patch from previous target position
        patch, _ = ut.get_patch(image, self.position, self.search_size)
        patch = np.multiply(patch, self.cosine_window)
        patch_fft = fft2(patch)

        # Localize the target - note that formula for localization uses FFT(H).conjugate(),
        # which we do keep in self.H
        response = ifft2(np.multiply(patch_fft, self.H))
        shift_y, shift_x = np.unravel_index(np.argmax(response), response.shape)

        if shift_x > self.search_size[0] / 2:
            shift_x -= self.search_size[0]
        if shift_y > self.search_size[1] / 2:
            shift_y -= self.search_size[1]


        # Update target position and template
        self.position = (self.position[0] + shift_x, self.position[1] + shift_y)
        self.target_patch, _ = ut.get_patch(image, self.position, self.search_size)
        self.target_patch = np.multiply(self.target_patch, self.cosine_window)

        # Update the filter
        new_H = self.compute_H(self.target_patch)
        self.H = (1 - self.alpha) * self.H + self.alpha * new_H

        # Return the position of the top left corner + width and height of target bounding box
        # Note that the size of the bounding box is not the same as the size of the search region
        # (unless self.enlarge_factor = 1)
        return [self.position[0] - self.bbox_size[0] / 2, self.position[1] - self.bbox_size[1] / 2, self.bbox_size[0], self.bbox_size[1]]

