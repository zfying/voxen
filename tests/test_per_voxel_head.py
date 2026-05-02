"""Unit tests for VPT per-voxel additions: PerVoxelLinearHead and arch_tag."""
import torch

from models.vpt import PerVoxelLinearHead
from staged import build_arch_tag


def test_per_voxel_head_forward_shape():
    head = PerVoxelLinearHead(feat_dim=8, n_voxels=17)
    feats = torch.randn(3, 17, 8)
    out = head(feats)
    assert out.shape == (3, 17)


def test_per_voxel_head_param_count():
    head = PerVoxelLinearHead(feat_dim=8, n_voxels=17)
    n = sum(p.numel() for p in head.parameters())
    assert n == 17 * 8 + 17


def test_per_voxel_head_per_voxel_independence():
    # Zeroing one voxel's weight row + bias should null only that voxel's output.
    head = PerVoxelLinearHead(feat_dim=4, n_voxels=5)
    with torch.no_grad():
        head.weight[2].zero_()
        head.bias[2].zero_()
    feats = torch.randn(2, 5, 4)
    out = head(feats)
    assert torch.all(out[:, 2] == 0)
    # Other voxels are not zero almost surely.
    assert not torch.all(out[:, 0] == 0)


def test_arch_tag_per_voxel_shared(make_args):
    args = make_args(
        encoder_arch='vpt', vpt_prompt_share='per_voxel', vpt_readout='linear',
        vpt_linear_feature='prompt', vpt_linear_share='shared',
        vpt_num_prompts_per_roi=1, readout_res='voxels',
    )
    assert build_arch_tag(args) == 'dinov2_q_vpt-per_voxel-linear-prompt-shared-K1'


def test_arch_tag_per_voxel_head(make_args):
    args = make_args(
        encoder_arch='vpt', vpt_prompt_share='per_voxel', vpt_readout='linear',
        vpt_linear_feature='prompt', vpt_linear_share='per_voxel',
        vpt_num_prompts_per_roi=3, readout_res='voxels',
    )
    assert build_arch_tag(args) == 'dinov2_q_vpt-per_voxel-linear-prompt-per_voxel-K3'
