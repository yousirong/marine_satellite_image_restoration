import torch
import torch.nn.functional as F
from torch import nn

class PartialConv2d(nn.Conv2d):
    def __init__(
        self, *args,
        multi_channel: bool = False,
        return_mask: bool = True,
        **kwargs
    ):
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

        # mask-updater 커널을 버퍼로 등록
        if self.multi_channel:
            upd = torch.ones(self.out_channels, self.in_channels, *self.kernel_size)
        else:
            upd = torch.ones(1, 1, *self.kernel_size)
        self.register_buffer('weight_mask_updater', upd)

        # sliding window 크기 (스칼라)
        self.slide_winsize = float(self.weight_mask_updater.numel())

    def forward(self, input: torch.Tensor, mask: torch.Tensor = None):
        batch, channel, height, width = input.shape
        device, dtype = input.device, input.dtype

        # 1) mask 준비
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

        # 2) update_mask 계산 (conv)
        update_mask = F.conv2d(
            mask,
            self.weight_mask_updater,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=1
        )

        # 3) 안전한 재정규화 인자 계산
        with torch.no_grad():
            eps = 1e-8
            ratio = self.slide_winsize / (update_mask + eps)
            mask_ratio = torch.where(update_mask > 0,
                                     ratio,
                                     torch.zeros_like(ratio))
        mask_ratio = mask_ratio.to(dtype)

        # update_mask을 0/1로 클램프
        updated_mask_clamped = (update_mask > 0).to(dtype)

        # 4) partial convolution 수행
        raw_out = super().forward(input * mask)

        # 5) bias 보정 및 mask 재적용
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
