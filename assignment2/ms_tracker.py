import numpy as np
import utils as ut


"""
A selected patch in the image is represented with a color histogram. For that,
we have utils.extract_histogram(patch, num_bins, weights) - the returned histogram
is NOT normalized!

q is the histogram of the template/target, it should be normalized.

We use kernels to smooth out the image, i.e. assign specific weights to each pixel, before
calculating the template histogram q (and later on p as well)

p is the histogram of the target candidate i.e. the current extracted patch; it should
also be normalized, like q

We need a similarity function for p and q, namely we use Bhattacharyya coefficient
(which is related to Hellinger divergence), which for p = {p_u}, q = {q_u}, u = 1, ..., m,
is equal to sum_{i=1 to m} sqrt(p_i*q_i). This is what we want to maximize! We do that
using mean shift. Read slides 39-45 for the formulas, but essentially, we linearized
the Bhattacharrya coefficient, and simplified the problem of maximizing it to maximizing
a KDE (as far as I'm concerned I'll call it a PDF) by applying mean shift! 

x_i and N are the same as for basic mean shift, w_i are given on the slides also (they come
from taking the derivative of the Bhattacharyya coefficient), and we still have the derivative
of the kernel in play. If we use Epanechnikov kernel, then this kernel derivative is again just 
1. In case we use Gaussian kernel, the derivative is still the Gaussian kernel.

Check out the implementation details on slides too.
"""
class MSTracker(ut.Tracker):
    def __init__(self, **params):
        self.kernel_type = params.get("kernel_type", "epanechnikov")  # specify kernel type
        self.sigma = params.get("sigma", 1)  # for Epanechnikov or Gaussian kernel
        self.num_bins = params.get("num_bins", 16)  # number of bins used for the target template/candidate histograms
        self.alpha = params.get("alpha", 0)  # for updating target template (so called update speed)
        self.eps_stop = params.get("eps_stop", 0.1)  # for stopping mean shift iterations
        self.eps_v = params.get("eps_v", 1e-4)  # for numerical stability when calculating weights v for mean shift


    def initialize(self, image, region):
        # param region - tuple of length 4; region[0] and region[1] are the x, y coordinates (respectively)
        # of the top left corner of the ground truth bbox. region[2] and region[3] are the width and height 
        # (respectively) of the ground truth bbox

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

        left = max(region[0], 0)
        top = max(region[1], 0)

        right = min(region[0] + region[2], image.shape[1] - 1)  # shape[1] is width i.e. num cols
        bottom = min(region[1] + region[3], image.shape[0] - 1)  # shape[0] is height i.e. num rows

        self.template = image[int(top):int(bottom), int(left):int(right)]  # initial target template i.e. "appearance"
        self.position = (region[0] + region[2] / 2, region[1] + region[3] / 2)  # coordinates of center of the template
        self.size = (region[2], region[3])  # width, height of the template

        if self.kernel_type == "epanechnikov":
            self.kernel = ut.epanechnikov_kernel(self.size[0], self.size[1], self.sigma)
            # self.kernel_deriv = np.copy(self.kernel)
            # self.kernel_deriv[self.kernel_deriv > 0] = 1.0  # uniform kernel
            self.kernel_deriv = np.ones_like(self.kernel)  # this should be right? cuz it doesn't change backprojected img
        elif self.kernel_type == "gaussian":
            self.kernel = ut.gaussian_kernel(self.size[0], self.size[1], self.sigma)
            self.kernel_deriv = self.kernel  # derivative of gaussian is gaussian
        else:
            raise ValueError(f"Unrecognized kernel type {self.kernel_type}")
        
        self.q = ut.extract_and_normalize_hist(self.template, self.num_bins, self.kernel)

        # Create these here, since they don't change across mean shift iterations (unless tracker is reinitalized)
        self.coords_x, self.coords_y = ut.coordinates_kernels(self.size)

    
    def mean_shift(self, image):
        curr_pos = self.position
        num_iters = 0
        while num_iters < 20:
            patch, _ = ut.get_patch(image, curr_pos, self.size)
            p = ut.extract_and_normalize_hist(patch, self.num_bins, self.kernel)
            v = np.sqrt(np.divide(self.q, p + self.eps_v))
            w = ut.backproject_histogram(patch, v, self.num_bins)

            w = np.multiply(w, self.kernel_deriv)

            change_x = np.divide(np.sum(np.multiply(self.coords_x, w)), np.sum(w))
            change_y = np.divide(np.sum(np.multiply(self.coords_y, w)), np.sum(w))
            if abs(change_x) < self.eps_stop and abs(change_y) < self.eps_stop:
                break

            curr_pos = (curr_pos[0] + change_x, curr_pos[1] + change_y)  # get_patch takes care of the rounding
            num_iters += 1

        self.position = (int(curr_pos[0]), int(curr_pos[1]))  # position of target (i.e. center of it) in next frame!


    def track(self, image):
        # Perform mean shift
        self.mean_shift(image)

        # Update target template and its histogram representation
        self.template, _ = ut.get_patch(image, self.position, self.size)
        q_new = ut.extract_and_normalize_hist(self.template, self.num_bins, self.kernel)
        self.q = (1 - self.alpha) * self.q + self.alpha * q_new

        # Return the position of the top left corner + width and height of predicted box
        return [self.position[0] - self.size[0] / 2, self.position[1] - self.size[1] / 2, self.size[0], self.size[1]]


if __name__ == "__main__":
    pass
