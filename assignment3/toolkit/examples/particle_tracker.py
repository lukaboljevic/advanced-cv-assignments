import numpy as np
import sympy as sp
from utils.my_utils import RW, NCV, NCA
from utils.tracker import Tracker

from assignment2.utils import epanechnikov_kernel, extract_and_normalize_hist, get_patch
from assignment4.utils import sample_gauss


class ParticleTracker(Tracker):
    def __init__(self, **params):
        self.motion_model = params.get("motion_model", NCV)  # motion model to use
        self.num_particles = params.get("num_particles", 100)  # number of particles
        self.q = params.get("q", 5)  # power spectral density of system covariance matrix Q
        self.alpha = params.get("alpha", 0)  # update speed
        self.sigma = params.get("sigma", 1)  # Sigma for Epanechnikov kernel
        self.dist_sigma = params.get("dist_sigma", 0.1)  # Sigma^2 for converting distance into probability
        self.num_bins = params.get("num_bins", 16)  # Number of bins when extracting histograms from patches

        self.Fi, self.Q = self.system_matrices()  # system matrix Fi and system covariance mtx Q
        self.dim = self.Fi.shape[0]  # number of dimensions of our state


    def name(self):
        mm = "RW" if self.motion_model == RW else "NCV" if self.motion_model == NCV else "NCA"
        return f"particle-{mm}mm-{self.num_particles}N-{self.q}q-{self.alpha}al-{self.dist_sigma}dsig-{self.num_bins}nbins"
    

    def system_matrices(self):
        """
        Calculate system matrices Fi (Φ) and Q (covariance) based on motion model.
        """
        T = sp.symbols("T")

        if self.motion_model == RW:
            # State is defined as X = [x, y]^T
            F = sp.Matrix([
                [0, 0],
                [0, 0]
            ])
            L = sp.Matrix([
                [1, 0],
                [0, 1]
            ])

        elif self.motion_model == NCV:
            # State is defined as X = [x, x', y, y']^T
            # x' and y' denote the velocities in x and y directions
            F = sp.Matrix([
                [0, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 0],
            ])
            L = sp.Matrix([
                [0, 0],
                [1, 0],
                [0, 0],
                [0, 1]
            ])

        elif self.motion_model == NCA:
            # State is defined as X = [x, x', x'', y, y', y'']^T
            # x' and y' denote the velocities in x and y directions
            # x'' and y'' denote the accelerations in x and y directions
            F = sp.Matrix([
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0]
            ])
            L = sp.Matrix([
                [0, 0],
                [0, 0],
                [1, 0],
                [0, 0],
                [0, 0],
                [0, 1]
            ])
        
        else:
            raise ValueError("Unsupported value for motion model.")

        # Calculate Fi, based on lecture slides
        Fi = sp.exp(F * T)

        # Calculate Q, based on lecture slides
        temp = Fi * L
        Q = sp.integrate(temp * self.q * temp.T, (T, 0, T))  # integrate w.r.t. T, from 0 to T

        # Replace T with 1 (1 since we're processing "frame-by-frame") and convert to numpy
        Fi = np.array(Fi.replace(T, 1).tolist(), dtype=np.float64)
        Q = np.array(Q.replace(T, 1).tolist(), dtype=np.float64)

        return Fi, Q
    

    def hellinger_distance(self, histogram):
        """
        Compute Hellinger distance between target histogram and histogram extracted
        at a given particle's position.
        """
        return np.sqrt(0.5 * ((np.sqrt(self.target_histogram) - np.sqrt(histogram))**2).sum())
        # Equivalent:
        # return 1 / np.sqrt(2) * np.linalg.norm(np.sqrt(self.target_histogram) - np.sqrt(histogram))


    def dist_to_prob(self, distance):
        """
        Convert distance to probability.
        """
        return np.exp(-0.5 * ((distance)**2 / (self.dist_sigma)))
    

    def normalize_particle_weights(self):
        """
        Normalize particle weights.
        """
        self.particle_weights = self.particle_weights / np.sum(self.particle_weights)


    def initialize(self, image, region):
        # Param region - tuple of length 4; region[0] and region[1] are the x, y coordinates 
        # (respectively) of the top left corner of the ground truth bbox. region[2] and region[3] 
        # are the width and height (respectively) of the ground truth bbox

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

        # Target bounding box position, size and image patch
        self.position = (region[0] + region[2] / 2, region[1] + region[3] / 2)  # x, y coordinates of center of the target
        self.bbox_size = (region[2], region[3])  # width, height of the target bounding box 
        self.target_patch, _ = get_patch(image, self.position, self.bbox_size)  # initial target template i.e. "appearance"

        # Epanechnikov kernel and target histogram
        self.kernel = epanechnikov_kernel(self.bbox_size[0], self.bbox_size[1], self.sigma)
        self.target_histogram = extract_and_normalize_hist(self.target_patch, self.num_bins, self.kernel)

        # Initialize particles and their weights
        mean = np.zeros(self.Q.shape[0])
        noise = sample_gauss(mean, self.Q, self.num_particles)
        self.particle_weights = np.ones(self.num_particles)
        self.normalize_particle_weights()
        self.particles = np.zeros((self.num_particles, self.dim))
        
        # Initial x coordinate of all particles is x coordinate of target center + noise
        # Similar idea applies to y coordinate
        # Based on motion model, the y coordinate is found in a different place in the particle state
        self.particles[:, 0] = self.position[0] + noise[:, 0]
        self.particles[:, self.motion_model] = self.position[1] + noise[:, self.motion_model]


    def track(self, image):
        # Resample particles
        weights_cumsum = np.cumsum(self.particle_weights)  # weights have to be normalized!
        rand_samples = np.random.rand(self.num_particles, 1)
        indices = np.digitize(rand_samples, weights_cumsum)
        self.particles = self.particles[indices.flatten(), :]

        # Move particles (deterministic shift + noise)
        mean = np.zeros(self.Q.shape[0])
        noise = sample_gauss(mean, self.Q, self.num_particles)
        self.particles = np.transpose(np.matmul(self.Fi, self.particles.transpose())) + noise

        # Clip x and y coordinates to be within the image. Otherwise, if the particle goes
        # out of bounds, get_patch returns a patch of other size, and then extract_histogram
        # fails.
        self.particles[:, 0] = np.clip(self.particles[:, 0], 0, image.shape[1] - 1)  # image.shape[1] is width
        self.particles[:, self.motion_model] = \
            np.clip(self.particles[:, self.motion_model], 0, image.shape[0] - 1)  # image.shape[0] is height

        # Recalculate particle weights
        for i, particle in enumerate(self.particles):
            patch, _ = get_patch(image, (particle[0], particle[self.motion_model]), self.bbox_size)
            patch_hist = extract_and_normalize_hist(patch, self.num_bins, self.kernel)
            hellinger = self.hellinger_distance(patch_hist)
            probability = self.dist_to_prob(hellinger)

            # New particle weight is exactly the calculated probability
            self.particle_weights[i] = probability

        # Normalize weights (imperative!)
        self.normalize_particle_weights()
            
        # Calculate new position of target as weighted mean of particle positions
        new_x = np.average(self.particles[:, 0], weights=self.particle_weights)
        new_y = np.average(self.particles[:, self.motion_model], weights=self.particle_weights)
        self.position = (new_x, new_y)

        # Update target histogram
        self.target_patch, _ = get_patch(image, self.position, self.bbox_size)
        new_histogram = extract_and_normalize_hist(self.target_patch, self.num_bins, self.kernel)
        self.target_histogram = (1 - self.alpha) * self.target_histogram + self.alpha * new_histogram

        # Return the position of the top left corner + width and height of target bounding box
        return [self.position[0] - self.bbox_size[0] / 2, self.position[1] - self.bbox_size[1] / 2, self.bbox_size[0], self.bbox_size[1]]

