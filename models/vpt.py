"""Visual Prompt Tuning components for the brain encoder.

Two pieces:
- PromptBank: learnable [N_ROI, K, D] tokens, one (or K) per ROI.
- PerROILinearHead: per-ROI Linear layers that project a per-ROI feature to
  only that ROI's vertices, scattered into the full hemisphere prediction tensor.
"""

import torch
from torch import nn


class PromptBank(nn.Module):
    def __init__(self, num_rois: int, num_prompts_per_roi: int, dim: int):
        super().__init__()
        self.num_rois = num_rois
        self.num_prompts_per_roi = num_prompts_per_roi
        self.dim = dim
        self.prompts = nn.Parameter(torch.zeros(num_rois, num_prompts_per_roi, dim))
        nn.init.trunc_normal_(self.prompts, std=0.02)

    def expand_for_batch(self, batch_size: int) -> torch.Tensor:
        # Returns [batch_size * num_rois, num_prompts_per_roi, dim] with ROI
        # varying fastest within each batch row, matching the image-replication
        # convention used in brain_encoder.forward (batch dim outer, ROI inner).
        p = self.prompts.unsqueeze(0).expand(batch_size, -1, -1, -1)
        return p.reshape(batch_size * self.num_rois, self.num_prompts_per_roi, self.dim)


class PerROILinearHead(nn.Module):
    """One Linear per ROI, mapping a [feat_dim] vector to that ROI's vertices.

    Outputs a tensor of shape [B, num_rois, hemi_vs] where each ROI's slot is
    populated only at the vertex indices belonging to that ROI; other indices
    are zero. The downstream SetCriterion then multiplies by the same ROI mask
    and sums across ROIs, giving per-vertex predictions.

    roi_masks: bool/0-1 tensor of shape [num_rois, hemi_vs]; row i is the mask
        over hemisphere vertices for ROI i (in the order returned by roi_masks()).
    """

    def __init__(self, feat_dim: int, roi_masks: torch.Tensor):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_rois, self.hemi_vs = roi_masks.shape

        self.heads = nn.ModuleList()
        index_buffers = []
        for i in range(self.num_rois):
            idx = torch.nonzero(roi_masks[i], as_tuple=False).flatten().long()
            self.heads.append(nn.Linear(feat_dim, idx.numel()))
            index_buffers.append(idx)

        # Store per-ROI vertex indices as buffers (move with .to(device) automatically).
        for i, idx in enumerate(index_buffers):
            self.register_buffer(f"roi_idx_{i}", idx, persistent=False)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: [B, num_rois, feat_dim]
        B = feats.shape[0]
        out = feats.new_zeros(B, self.num_rois, self.hemi_vs)
        for i, head in enumerate(self.heads):
            idx = getattr(self, f"roi_idx_{i}")
            if idx.numel() == 0:
                continue
            out[:, i].index_copy_(1, idx, head(feats[:, i]))
        return out


class PerVoxelLinearHead(nn.Module):
    """One D->1 linear per voxel, computed in a single fused op.

    Used by VPT per-voxel mode: each voxel has its own readout weight + bias,
    consuming a per-voxel feature vector (typically the mean-pooled K prompt
    tokens after a per-voxel backbone forward).
    """

    def __init__(self, feat_dim: int, n_voxels: int):
        super().__init__()
        self.feat_dim = feat_dim
        self.n_voxels = n_voxels
        self.weight = nn.Parameter(torch.empty(n_voxels, feat_dim))
        self.bias = nn.Parameter(torch.zeros(n_voxels))
        nn.init.trunc_normal_(self.weight, std=0.02)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: [B, n_voxels, feat_dim] -> [B, n_voxels]
        return (feats * self.weight).sum(dim=-1) + self.bias
