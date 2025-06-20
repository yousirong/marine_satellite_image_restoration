import torch
import torch.nn as nn
import torch.nn.functional as F

class KnowledgeConsistentAttention(nn.Module):
    def __init__(self, patch_size=3, propagate_size=3, stride=1):
        super().__init__()
        self.patch_size = patch_size
        self.propagate_size = propagate_size
        self.stride = stride

        # 저장용 (detached) 이전 attention/mask
        self.att_scores_prev = None  # (B, patches, H, W)
        self.masks_prev = None       # (B, 1, H, W)

        # 학습 파라미터
        self.ratio = nn.Parameter(torch.ones(1))

    def forward(self, foreground: torch.Tensor, masks: torch.Tensor):
        B, C, H, W = foreground.shape
        device, dtype = foreground.device, foreground.dtype

        # 1) mask 크기 보정
        if masks.shape[-2:] != (H, W):
            masks = F.interpolate(masks, size=(H, W), mode='nearest')
        masks = masks.to(device=device, dtype=dtype)

        # 2) 패치별 커널 생성
        # conv_kernels_all: (B, patches, C, 1, 1)
        patches = H * W
        conv_kernels_all = (
            foreground.view(B, C, patches, 1, 1)
                      .permute(0, 2, 1, 3, 4)
        )

        outputs = []
        new_att_list = []

        for i in range(B):
            x = foreground[i:i+1]            # (1, C, H, W)
            kernels = conv_kernels_all[i]     # (patches, C, 1, 1)

            # normalize kernels
            norm = kernels.flatten(1).norm(dim=1, keepdim=True).view(-1,1,1,1)
            kernels = kernels / (norm + 1e-8)

            # 3) 유사도 계산
            conv_res = F.conv2d(x, kernels, padding=self.patch_size//2)

            # 4) propagate (POOL)
            if self.propagate_size > 1:
                k = self.propagate_size
                conv_res = F.avg_pool2d(conv_res, kernel_size=k, stride=1, padding=k//2) * (k*k)

            # 5) attention scores
            attn = F.softmax(conv_res, dim=1)  # expect 4D: (1, patches, H, W)

            # 6) 이전과 혼합
            if self.att_scores_prev is not None:
                prev_a = self.att_scores_prev[i:i+1]  # (1, patches, H, W)
                prev_m = self.masks_prev[i:i+1]       # (1, 1, H, W)
                alpha = (torch.abs(self.ratio) + 1e-7)
                attn = (prev_a * prev_m + attn * alpha) / (prev_m + alpha)

            # ensure 4D (avoid stray dims)
            if attn.dim() == 5:
                attn = attn.squeeze(1)

            # 7) 복원
            out = F.conv_transpose2d(
                attn, kernels,
                stride=1, padding=self.patch_size//2
            )  # (1, C, H, W)

            outputs.append(out)
            new_att_list.append(attn)

        # 8) 이전값 업데이트 (detach)
        self.att_scores_prev = torch.cat(new_att_list, dim=0).detach()
        self.masks_prev = masks.view(B, 1, H, W).detach()

        # 9) 결과 반환
        return torch.cat(outputs, dim=0)  # (B, C, H, W)


class AttentionModule(nn.Module):
    def __init__(self, inchannel, patch_size_list=[1], propagate_size_list=[3], stride_list=[1]):
        super().__init__()
        assert len(patch_size_list) == len(propagate_size_list) == len(stride_list), \
            "patch_size_list, propagate_size_list, stride_list must match lengths"

        self.att = KnowledgeConsistentAttention(
            patch_size=patch_size_list[0],
            propagate_size=propagate_size_list[0],
            stride=stride_list[0]
        )
        self.combiner = nn.Conv2d(inchannel * 2, inchannel, kernel_size=1)

    def forward(self, foreground: torch.Tensor, mask: torch.Tensor):
        attended = self.att(foreground, mask)        # (B, C, H, W)
        combined = torch.cat([attended, foreground], dim=1)
        return self.combiner(combined)               # (B, C, H, W)
