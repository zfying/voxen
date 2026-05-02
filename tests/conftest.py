"""Shared fixtures for staged.py unit tests.

Tiny in-memory models / arg objects so the suite needs no NSD data, no GPU,
no DINOv2 download. Run with `pytest tests/ -q` from the repo root.
"""
import os
import sys
import types

import pytest
import torch
from torch import nn

# Make the repo root importable when invoked as `pytest tests/`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TinyVPTModel(nn.Module):
    """Mimics the parameter-name layout of brain_encoder for the VPT path.

    No real forward; only `named_parameters()` is exercised by the helpers.
    """
    def __init__(self, *, with_per_roi=False, with_shared_prompts=False, K=2, D=4, hidden=4, lh_vs=6, rh_vs=5, n_rois=3):
        super().__init__()
        self.backbone_model = nn.Linear(D, D)            # always frozen
        if with_shared_prompts:
            self.shared_prompts = nn.Parameter(torch.zeros(K, D))
        else:
            # Mirror PromptBank: a Module owning a Parameter named 'prompts'.
            self.prompt_bank = nn.Module()
            self.prompt_bank.prompts = nn.Parameter(torch.zeros(n_rois, K, D))
            # nn.Module needs explicit registration:
            self.prompt_bank.register_parameter('prompts', self.prompt_bank.prompts)
        self.transformer = nn.Linear(D, hidden)
        self.query_embed = nn.Embedding(n_rois, hidden)
        self.lh_embed = nn.Sequential(nn.Linear(hidden, lh_vs))
        self.rh_embed = nn.Sequential(nn.Linear(hidden, rh_vs))
        if with_per_roi:
            self.lh_per_roi_head = nn.Linear(D, lh_vs)
            self.rh_per_roi_head = nn.Linear(D, rh_vs)


@pytest.fixture
def tiny_vpt_model():
    return TinyVPTModel()


@pytest.fixture
def tiny_vpt_model_shared():
    return TinyVPTModel(with_shared_prompts=True)


@pytest.fixture
def tiny_vpt_model_per_roi():
    return TinyVPTModel(with_per_roi=True)


def _make_args(**overrides):
    """Build an argparse-like Namespace with defaults matching staged.py needs."""
    ns = types.SimpleNamespace(
        encoder_arch='vpt',
        backbone_arch='dinov2_q',
        vpt_prompt_share='shared',
        vpt_readout='decoder',
        vpt_linear_feature='prompt',
        vpt_linear_share='shared',
        vpt_num_prompts_per_roi=5,
        vpt_decoder_attend_prompts=False,
        vpt_staged=False,
        vpt_load_readout='auto',
        subj=1,
        run=1,
        readout_res='rois_all',
        enc_output_layer=1,
        output_path='./results/',
    )
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


@pytest.fixture
def make_args():
    return _make_args


@pytest.fixture
def tmp_results_tree(tmp_path):
    """Create a fake results/ tree with a baseline transformer checkpoint."""
    output_path = str(tmp_path) + '/'
    base = os.path.join(
        output_path, 'nsd_test', 'dinov2_q_transformer',
        'subj_01', 'rois_all', 'enc_1', 'run_1',
    )
    os.makedirs(base, exist_ok=True)
    ckpt_path = os.path.join(base, 'checkpoint.pth')
    # Stash a tiny dummy checkpoint with a couple of recognisable keys.
    torch.save({
        'model': {
            'lh_embed.0.weight': torch.zeros(6, 4),
            'lh_embed.0.bias': torch.zeros(6),
            'rh_embed.0.weight': torch.zeros(5, 4),
            'rh_embed.0.bias': torch.zeros(5),
            'transformer.weight': torch.zeros(4, 4),
            'transformer.bias': torch.zeros(4),
            'query_embed.weight': torch.zeros(3, 4),
        },
        'args': types.SimpleNamespace(val_perf=0.123),
        'val_perf': 0.123,
    }, ckpt_path)
    return {'output_path': output_path, 'ckpt_path': ckpt_path}
