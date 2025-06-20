###############################################################################
# BSD 3-Clause License
#
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Author & Contact: Guilin Liu (guilinl@nvidia.com)
###############################################################################

import torch
import torch.nn.functional as F
from torch import nn


class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, multi_channel=False, return_mask=True, **kwargs):
        """
        Partial convolution layer: masks invalid (missing) regions in the input
        and renormalizes.
        Args:
            multi_channel (bool): if True, mask has same channels as input
            return_mask (bool): if True, forward returns (output, updated_mask)
        """
        super().__init__(*args, **kwargs)
        self.multi_channel = multi_channel
        self.return_mask = return_mask

        # initialize the mask-updating kernel
        if self.multi_channel:
            self.weight_mask_updater = torch.ones(
                self.out_channels, self.in_channels, *self.kernel_size
            )
        else:
            self.weight_mask_updater = torch.ones(
                1, 1, *self.kernel_size
            )

        # total number of pixels in sliding window (scalar)
        self.slide_winsize = float(self.weight_mask_updater.numel())

    def forward(self, input: torch.Tensor, mask: torch.Tensor = None):
        batch, channel, height, width = input.shape
        device, dtype = input.device, input.dtype

        # prepare mask
        if mask is None:
            mask = torch.ones(
                batch,
                channel if self.multi_channel else 1,
                height,
                width,
                device=device,
                dtype=dtype
            )
        else:
            mask = mask.to(device=device, dtype=dtype)

        # ensure kernel on correct device/dtype
        if self.weight_mask_updater.device != device or self.weight_mask_updater.dtype != dtype:
            self.weight_mask_updater = self.weight_mask_updater.to(device=device, dtype=dtype)

        # 1) compute updated mask via convolution
        update_mask = F.conv2d(
            mask,
            self.weight_mask_updater,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=1
        )

        # 2) compute renormalization factor safely (no grad, local var)
        with torch.no_grad():
            # only compute ratio where update_mask > 0 to avoid inf*0 -> nan
            ratio = self.slide_winsize / (update_mask + 1e-8)
            mask_ratio = torch.where(update_mask > 0,
                                     ratio,
                                     torch.zeros_like(ratio))
        mask_ratio = mask_ratio.to(dtype)

        # clamp update_mask to {0,1}
        updated_mask_clamped = (update_mask > 0).to(dtype)

        # 3) apply partial convolution to masked input
        raw_out = super().forward(input * mask)

        # 4) re-normalize output and re-apply mask
        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = (raw_out - bias_view) * mask_ratio + bias_view
            output = output * updated_mask_clamped
        else:
            output = raw_out * mask_ratio

        if self.return_mask:
            return output, updated_mask_clamped
        else:
            return output
