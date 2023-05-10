from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import cv2
import sys
import os
from math import ceil
from collections import namedtuple
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from got10k.trackers import Tracker

from . import ops
from .backbones import AlexNetV1
from .heads import SiamFC
from .losses import BalancedLoss
from .datasets import Pair
from .transforms import SiamFCTransforms


__all__ = ['TrackerSiamFC']


class Net(nn.Module):

    def __init__(self, backbone, head):
        super(Net, self).__init__()
        self.backbone = backbone
        self.head = head
    
    def forward(self, z, x):
        z = self.backbone(z)
        x = self.backbone(x)
        return self.head(z, x)


class TrackerSiamFC(Tracker):

    def __init__(self, net_path=None, long_term=False, long_term_params=None, **kwargs):
        super(TrackerSiamFC, self).__init__('SiamFC', True)
        self.cfg = self.parse_args(**kwargs)

        # setup GPU device if available
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        # setup model
        self.net = Net(
            backbone=AlexNetV1(),
            head=SiamFC(self.cfg.out_scale))
        ops.init_weights(self.net)
        
        # load checkpoint if provided
        if net_path is not None:
            self.net.load_state_dict(torch.load(
                net_path, map_location=lambda storage, loc: storage))
        self.net = self.net.to(self.device)

        # Luka: whether to perform long term tracking instead of short term
        self.long_term = long_term
        self.long_term_params = long_term_params
        self.redetecting = False
        self.frames_in_redetection = 0

        # setup criterion
        self.criterion = BalancedLoss()

        # setup optimizer
        self.optimizer = optim.SGD(
            self.net.parameters(),
            lr=self.cfg.initial_lr,
            weight_decay=self.cfg.weight_decay,
            momentum=self.cfg.momentum)
        
        # setup lr scheduler
        gamma = np.power(
            self.cfg.ultimate_lr / self.cfg.initial_lr,
            1.0 / self.cfg.epoch_num)
        self.lr_scheduler = ExponentialLR(self.optimizer, gamma)


    def parse_args(self, **kwargs):
        # default parameters
        cfg = {
            # basic parameters
            'out_scale': 0.001,
            'exemplar_sz': 127,
            'instance_sz': 255,
            'context': 0.5,
            # inference parameters
            'scale_num': 3,
            'scale_step': 1.0375,
            'scale_lr': 0.59,
            'scale_penalty': 0.9745,
            'window_influence': 0.176,
            'response_sz': 17,
            'response_up': 16,
            'total_stride': 8,
            # train parameters
            'epoch_num': 50,
            'batch_size': 8,
            'num_workers': 16,  # 32
            'initial_lr': 1e-2,
            'ultimate_lr': 1e-5,
            'weight_decay': 5e-4,
            'momentum': 0.9,
            'r_pos': 16,
            'r_neg': 0}
        
        for key, val in kwargs.items():
            if key in cfg:
                cfg.update({key: val})
        return namedtuple('Config', cfg.keys())(**cfg)


    @torch.no_grad()
    def init(self, img, box):
        # set to evaluation mode
        self.net.eval()

        # convert box to 0-indexed and center based [y, x, h, w]
        box = np.array([
            box[1] - 1 + (box[3] - 1) / 2,
            box[0] - 1 + (box[2] - 1) / 2,
            box[3], box[2]], dtype=np.float32)
        self.center, self.target_sz = box[:2], box[2:]

        # create hanning window
        self.upscale_sz = self.cfg.response_up * self.cfg.response_sz
        self.hann_window = np.outer(
            np.hanning(self.upscale_sz),
            np.hanning(self.upscale_sz))
        self.hann_window /= self.hann_window.sum()

        # search scale factors
        self.scale_factors = self.cfg.scale_step ** np.linspace(
            -(self.cfg.scale_num // 2),
            self.cfg.scale_num // 2, self.cfg.scale_num)

        # exemplar and search sizes
        context = self.cfg.context * np.sum(self.target_sz)
        self.z_sz = np.sqrt(np.prod(self.target_sz + context))
        self.x_sz = self.z_sz * \
            self.cfg.instance_sz / self.cfg.exemplar_sz
        
        # exemplar image
        self.avg_color = np.mean(img, axis=(0, 1))
        z = ops.crop_and_resize(
            img, self.center, self.z_sz,
            out_size=self.cfg.exemplar_sz,
            border_value=self.avg_color)
        
        # exemplar features
        z = torch.from_numpy(z).to(
            self.device).permute(2, 0, 1).unsqueeze(0).float()
        self.kernel = self.net.backbone(z)


    @torch.no_grad()
    def update(self, img):
        img_height, img_width = img.shape[:2]
        redetection_bboxes = None
        redetection_centers = None

        # set to evaluation mode
        self.net.eval()

        if self.long_term and self.redetecting:
            self.frames_in_redetection += 1
            if self.frames_in_redetection >= 35:
                # ad hoc solution, doesn't affect performance THAT much across the entire
                # dataset but helps in some individual sequences and increases recall by ~1.5-2%
                self.center = np.array([img_height // 2, img_width // 2])

            # Luka: we want to redetect the target at fixed scale, but at multiple positions.
            # Generate a number of locations i.e. center coordinates as y, x in a numpy array - same as self.center
            increased_size = self.x_sz * self.long_term_params["scale_up"]

            if self.long_term_params["sampling"] == "uniform":
                lower_x = lower_y = 0 + increased_size // 2
                upper_x = img_width - increased_size // 2
                upper_y = img_height - increased_size // 2
                redetection_centers = [
                    np.array([
                        np.random.uniform(lower_y, upper_y), np.random.uniform(lower_x, upper_x)
                    ]) for _ in range(self.long_term_params["num_locations"])
                ]
            elif self.long_term_params["sampling"] == "gaussian":
                # "Hacky" way to compute deviation based on size of target and number of frames since redetection started
                deviation = min(int((increased_size * 0.7)) + int(ceil(1.1**self.frames_in_redetection)), 200)  # cap to 200
                num_locations = min(self.long_term_params["num_locations"] + self.frames_in_redetection // 3, 25)  # cap to 25 locations
                redetection_centers = np.random.normal(self.center, deviation, (num_locations, 2))
            else:
                raise ValueError("Unsupported sampling technique")

            # 1-indexed and left corner based bounding boxes
            redetection_bboxes = [np.array([
                center[1] + 1 - (increased_size - 1) / 2,
                center[0] + 1 - (increased_size - 1) / 2,
                increased_size,
                increased_size
                ]) for center in redetection_centers
            ]

            # Multiple images at fixed scale
            x = [
                ops.crop_and_resize(img, center, increased_size, out_size=self.cfg.instance_sz, border_value=self.avg_color)
                for center in redetection_centers
            ]
        
        else:
            # Search images; Luka: target is searched at multiple scales around same starting positions
            x = [ops.crop_and_resize(
                img, self.center, self.x_sz * f,
                out_size=self.cfg.instance_sz,
                border_value=self.avg_color) for f in self.scale_factors]
            
        x = np.stack(x, axis=0)
        x = torch.from_numpy(x).to(
            self.device).permute(0, 3, 1, 2).float()
        
        # responses
        x = self.net.backbone(x)
        responses = self.net.head(self.kernel, x)
        responses = responses.squeeze(1).cpu().numpy()

        # upsample responses and penalize scale changes, [Luka] if we're not in redetection stage
        responses = np.stack([cv2.resize(
            u, (self.upscale_sz, self.upscale_sz),
            interpolation=cv2.INTER_CUBIC)
            for u in responses])
        if not (self.long_term and self.redetecting):
            responses[:self.cfg.scale_num // 2] *= self.cfg.scale_penalty
            responses[self.cfg.scale_num // 2 + 1:] *= self.cfg.scale_penalty

        # peak scale
        scale_id = np.argmax(np.amax(responses, axis=(1, 2)))

        # peak location
        response = responses[scale_id]
        max_resp = max(0, response.max())
        if self.long_term and not self.redetecting:
            if max_resp < self.long_term_params["failure_threshold"]:
                self.redetecting = True
                # Luka: If the target is lost in current frame, we immediately want to start redetecting, 
                # mostly for the sake of not updating the tracker
                return self.update(img)
        
        if self.long_term and self.redetecting:
            # Luka: set center to one of the generated centers, and DON'T UPDATE TARGET SIZE!
            if self.long_term_params["sampling"] == "uniform":
                self.center = redetection_centers[scale_id]
            # if gaussian, we update the center ONLY when we have redetected the target
        else:
            # peak location
            response -= response.min()
            response /= response.sum() + 1e-16
            response = (1 - self.cfg.window_influence) * response + \
                self.cfg.window_influence * self.hann_window
            loc = np.unravel_index(response.argmax(), response.shape)

            # locate target center
            disp_in_response = np.array(loc) - (self.upscale_sz - 1) / 2
            disp_in_instance = disp_in_response * \
                self.cfg.total_stride / self.cfg.response_up
            disp_in_image = disp_in_instance * self.x_sz * \
                self.scale_factors[scale_id] / self.cfg.instance_sz
            self.center += disp_in_image

            # update target size
            scale =  (1 - self.cfg.scale_lr) * 1.0 + \
                self.cfg.scale_lr * self.scale_factors[scale_id]
            self.target_sz *= scale
            self.z_sz *= scale
            self.x_sz *= scale

        # 1-indexed and left-top based predicted bounding box (most probable target location)
        box = np.array([
            self.center[1] + 1 - (self.target_sz[1] - 1) / 2,
            self.center[0] + 1 - (self.target_sz[0] - 1) / 2,
            self.target_sz[1], self.target_sz[0]])
        
        if self.long_term:
            # The "issue" with this is that on the exact frame when we redetect the target, i.e. when redetection 
            # stage stops, we will not report the exact position, but a slightly offset one, coming from one of the 
            # generated "redetection centers". In the next frame, the SiamFC tracker is capable of adjusting this 
            # offset position to the right one.
            self.redetecting = (max_resp < self.long_term_params["failure_threshold"])
            if not self.redetecting:
                self.frames_in_redetection = 0
                if redetection_centers is not None and self.long_term_params["sampling"] == "gaussian":
                    self.center = redetection_centers[scale_id]

        return box, max_resp, redetection_bboxes


    def train_step(self, batch, backward=True):
        # set network mode
        self.net.train(backward)

        # parse batch data
        z = batch[0].to(self.device, non_blocking=self.cuda)
        x = batch[1].to(self.device, non_blocking=self.cuda)

        with torch.set_grad_enabled(backward):
            # inference
            responses = self.net(z, x)

            # calculate loss
            labels = self._create_labels(responses.size())
            loss = self.criterion(responses, labels)
            
            if backward:
                # back propagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        return loss.item()


    @torch.enable_grad()
    def train_over(self, seqs, val_seqs=None,
                   save_dir='pretrained'):
        # set to train mode
        self.net.train()

        # create save_dir folder
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # setup dataset
        transforms = SiamFCTransforms(
            exemplar_sz=self.cfg.exemplar_sz,
            instance_sz=self.cfg.instance_sz,
            context=self.cfg.context)
        dataset = Pair(
            seqs=seqs,
            transforms=transforms)
        
        # setup dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cuda,
            drop_last=True)
        
        # loop over epochs
        for epoch in range(self.cfg.epoch_num):
            # update lr at each epoch
            self.lr_scheduler.step(epoch=epoch)

            # loop over dataloader
            for it, batch in enumerate(dataloader):
                loss = self.train_step(batch, backward=True)
                print('Epoch: {} [{}/{}] Loss: {:.5f}'.format(
                    epoch + 1, it + 1, len(dataloader), loss))
                sys.stdout.flush()
            
            # save checkpoint
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            net_path = os.path.join(
                save_dir, 'siamfc_alexnet_e%d.pth' % (epoch + 1))
            torch.save(self.net.state_dict(), net_path)


    def _create_labels(self, size):
        # skip if same sized labels already created
        if hasattr(self, 'labels') and self.labels.size() == size:
            return self.labels

        def logistic_labels(x, y, r_pos, r_neg):
            dist = np.abs(x) + np.abs(y)  # block distance
            labels = np.where(dist <= r_pos,
                              np.ones_like(x),
                              np.where(dist < r_neg,
                                       np.ones_like(x) * 0.5,
                                       np.zeros_like(x)))
            return labels

        # distances along x- and y-axis
        n, c, h, w = size
        x = np.arange(w) - (w - 1) / 2
        y = np.arange(h) - (h - 1) / 2
        x, y = np.meshgrid(x, y)

        # create logistic labels
        r_pos = self.cfg.r_pos / self.cfg.total_stride
        r_neg = self.cfg.r_neg / self.cfg.total_stride
        labels = logistic_labels(x, y, r_pos, r_neg)

        # repeat to size
        labels = labels.reshape((1, 1, h, w))
        labels = np.tile(labels, (n, c, 1, 1))

        # convert to tensors
        self.labels = torch.from_numpy(labels).to(self.device).float()
        
        return self.labels
