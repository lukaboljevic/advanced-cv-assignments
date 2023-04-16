import math
import numpy as np
import cv2


def kalman_step(A, C, Q, R, y, x, V):
    """
    Perform Kalman filter prediction and update.

    Parameters
    ----------
    A : system matrix (on lecture slides, Fi or Φ)
    C : observation matrix (on lecture slides, H)
    Q : system covariance (on lecture slides, Q)
    R : observation covariance (on lecture slides, R)
    y(:) : observation at time t (on lecture slides, y_k)
    x(:) : E[X | y(:, 1:t-1)], prior mean (on lecture slides, x_k)
    V(:,:) : Cov[X | y(:, 1:t-1)], prior covariance (on lecture slides, P_k)

    Returns (X is the hidden state being estimated)
    -----------------------------------------------
    xnew(:) : E[ X | y(:, 1:t) ], new state
    Vnew(:,:) : Var[ X(t) | y(:, 1:t) ], new state's covariance
    VVnew(:,:) : Cov[ X(t), X(t-1) | y(:, 1:t) ]
    loglik : log P(y(:,t) | y(:,1:t-1)), log-likelihood of innovation
    """

    # Prediction (state and covariance)
    xpred = np.matmul(A, x)
    Vpred = np.matmul(np.matmul(A, V), A.transpose()) + Q

    # Kalman gain matrix
    ss = max(V.shape)
    S = np.matmul(np.matmul(C, Vpred), C.transpose()) + R
    K = np.linalg.lstsq(S.transpose(), np.matmul(Vpred, C.transpose()).transpose(), rcond=None)[0].transpose()
    # If there is no observation vector, set K = zeros(ss).

    # Error (innovation)
    e = y - np.matmul(C, xpred)
    loglik = gaussian_prob(e, np.zeros((1, e.size), dtype=np.float32), S, use_log=True)

    # Update (new state and its new covariance)
    xnew = xpred + np.matmul(K, e)
    Vnew = np.matmul((np.eye(ss) - np.matmul(K, C)), Vpred)
    VVnew = np.matmul(np.matmul((np.eye(ss) - np.matmul(K, C)), A), V)\
    
    return xnew, Vnew, loglik, VVnew


def gaussian_prob(x, m, C, use_log=False):
    """
    Evaluate multivariate Gaussian density.

    p(i) = N(X(:,i), m, C) where C = covariance matrix and each COLUMN of x is a datavector
    p = gaussian_prob(X, m, C, 1) returns log N(X(:,i), m, C) (to prevent underflow).
    If X has size dxN, then p has size Nx1, where N = number of examples
    """
    
    if m.size == 1:
        x = x.flatten().transpose()

    d, N = x.shape

    m = m.flatten()
    M = np.reshape(m * np.ones(m.shape, dtype=np.float32), x.shape)
    denom = (2 * math.pi)**(d/2) * np.sqrt(np.abs(np.linalg.det(C)))
    mahal = np.sum(np.linalg.solve(C.transpose(), (x - M)) * (x - M))   # Chris Bregler's trick

    if np.any(mahal < 0):
        print('Warning: mahal < 0 => C is not psd')

    if use_log:
        p = -0.5 * mahal - np.log(denom)
    else:
        p = np.divide(np.exp(-0.5 * mahal), (denom + 1e-20))

    return p


def sample_gauss(mu, sigma, n):
    """
    Sample n samples from a multivariate normal distribution.
    """
    return np.random.multivariate_normal(mu, sigma, n)


def get_patch(img, center, size):
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


# base class for tracker
class Tracker():
    def __init__(self, params):
        self.parameters = params

    def initialize(self, image, region):
        raise NotImplementedError

    def track(self, image):
        raise NotImplementedError
