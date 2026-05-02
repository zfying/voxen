"""Helpers for 3-stage VPT training.

Pure functions (no I/O at import time, no CUDA), so the test suite can import
them without an NSD dataset or a GPU.
"""
from __future__ import annotations

import os
from typing import Iterable, Optional

import torch
from torch import nn


# ---------------------------------------------------------------------------
# Param partitioning
# ---------------------------------------------------------------------------

_PROMPT_PREFIXES = ('prompt_bank', 'shared_prompts')
_BACKBONE_PREFIX = 'backbone_model'


def _is_backbone(name: str) -> bool:
    return name.startswith(_BACKBONE_PREFIX) or f'.{_BACKBONE_PREFIX}' in name


def _is_prompt(name: str) -> bool:
    return any(name.startswith(p) or name == p for p in _PROMPT_PREFIXES)


def vpt_param_groups(model: nn.Module) -> dict:
    """Partition model params into {'prompts', 'readout'}, excluding backbone.

    Returned values are lists of (name, parameter) tuples.
    """
    prompts, readout = [], []
    for n, p in model.named_parameters():
        if _is_backbone(n):
            continue
        if _is_prompt(n):
            prompts.append((n, p))
        else:
            readout.append((n, p))
    return {'prompts': prompts, 'readout': readout}


def set_stage(model: nn.Module, stage: int) -> dict:
    """Toggle requires_grad for the 3-stage curriculum.

    stage 1: readout only      | stage 2: prompts only | stage 3: both
    Backbone params stay frozen regardless. Returns a dict of trainable param
    counts for logging.
    """
    if stage not in (1, 2, 3):
        raise ValueError(f'stage must be 1, 2, or 3; got {stage}')

    groups = vpt_param_groups(model)
    train_readout = stage in (1, 3)
    train_prompts = stage in (2, 3)

    for _, p in groups['readout']:
        p.requires_grad = train_readout
    for _, p in groups['prompts']:
        p.requires_grad = train_prompts

    # Backbone always frozen.
    for n, p in model.named_parameters():
        if _is_backbone(n):
            p.requires_grad = False

    return {
        'stage': stage,
        'readout_params': sum(p.numel() for _, p in groups['readout']) if train_readout else 0,
        'prompt_params': sum(p.numel() for _, p in groups['prompts']) if train_prompts else 0,
        'readout_tensors': len(groups['readout']) if train_readout else 0,
        'prompt_tensors': len(groups['prompts']) if train_prompts else 0,
    }


# ---------------------------------------------------------------------------
# Partial state-dict loading
# ---------------------------------------------------------------------------


def partial_load_state_dict(
    model: nn.Module,
    src_state_dict: dict,
    prefixes_to_skip: Iterable[str] = (_BACKBONE_PREFIX,),
) -> dict:
    """Load only keys present in both, with matching shapes.

    Skips keys whose name starts with any of `prefixes_to_skip`. Does not raise
    on shape mismatch or missing/extra keys; returns lists for inspection.
    """
    skip = tuple(prefixes_to_skip)
    dst = model.state_dict()
    to_load = {}
    skipped = []
    for k, v in src_state_dict.items():
        if any(k.startswith(s) for s in skip):
            skipped.append((k, 'prefix-skipped'))
            continue
        if k not in dst:
            skipped.append((k, 'missing-in-dest'))
            continue
        if dst[k].shape != v.shape:
            skipped.append((k, f'shape-mismatch: dst={tuple(dst[k].shape)} src={tuple(v.shape)}'))
            continue
        to_load[k] = v

    missing_in_src = [k for k in dst.keys() if k not in src_state_dict and not any(k.startswith(s) for s in skip)]
    model.load_state_dict(to_load, strict=False)
    return {
        'loaded': sorted(to_load.keys()),
        'skipped': skipped,
        'missing_in_src': missing_in_src,
    }


# ---------------------------------------------------------------------------
# arch_tag (output dir name)
# ---------------------------------------------------------------------------


def build_arch_tag(args) -> str:
    """Reproduce main.py's arch_tag rule, plus '-staged' suffix when staged."""
    if args.encoder_arch == 'vpt':
        tag = (
            f'{args.backbone_arch}_vpt-{args.vpt_prompt_share}-{args.vpt_readout}'
            f'-{args.vpt_linear_feature}-{args.vpt_linear_share}'
            f'-K{args.vpt_num_prompts_per_roi}'
        )
        if getattr(args, 'vpt_decoder_attend_prompts', False):
            tag += '-attP'
    else:
        tag = f'{args.backbone_arch}_{args.encoder_arch}'

    if getattr(args, 'vpt_staged', False):
        tag += '-staged'
    return tag


# ---------------------------------------------------------------------------
# Baseline checkpoint resolution for stage-1 auto-load
# ---------------------------------------------------------------------------


def _baseline_arch_tag_for_vpt(args) -> Optional[str]:
    """Return the arch_tag of a non-VPT baseline whose readout is shape-
    compatible with this VPT model, or None if no compatible baseline exists.
    """
    if args.encoder_arch != 'vpt':
        return None
    if args.vpt_readout == 'decoder':
        return f'{args.backbone_arch}_transformer'
    # Linear readouts: VPT uses backbone_dim->vertices; the linear baseline
    # uses Conv2d+flatten (different feature dim). Per-ROI heads have unique
    # parameter names. No compatible baseline.
    return None


def resolve_baseline_checkpoint(args) -> Optional[str]:
    """Decide where stage 1 should try to load a readout from.

    Honours --vpt_load_readout: 'none' -> None; explicit path -> that path
    (verbatim); 'auto' -> derived path under args.output_path if it exists.
    """
    mode = getattr(args, 'vpt_load_readout', 'auto')
    if mode == 'none':
        return None
    if mode != 'auto':
        # Explicit path. Caller checks existence and partial-loads.
        return mode

    base_tag = _baseline_arch_tag_for_vpt(args)
    if base_tag is None:
        return None

    # Subj is formatted as zero-padded 2 digits in main; do the same here.
    subj = format(int(args.subj), '02') if not isinstance(args.subj, str) else args.subj.zfill(2)
    output_path = args.output_path if args.output_path.endswith('/') else args.output_path + '/'
    path = (
        f'{output_path}nsd_test/{base_tag}/subj_{subj}/'
        f'{args.readout_res}/enc_{args.enc_output_layer}/run_{args.run}/checkpoint.pth'
    )
    return path if os.path.exists(path) else None


# ---------------------------------------------------------------------------
# Per-epoch log row
# ---------------------------------------------------------------------------

LOG_HEADER = 'epoch\tstage\ttrain_loss\tval_loss\tval_perf\tbest'


def format_epoch_log_row(
    epoch: int,
    stage: int,
    train_loss: float,
    val_loss: float,
    val_perf: float,
    is_best: bool,
) -> str:
    best = '*' if is_best else ''
    return (
        f'{epoch}\t{stage}\t'
        f'{train_loss:.6g}\t{val_loss:.6g}\t{val_perf:.6g}\t{best}'
    )


def parse_epoch_log_row(row: str) -> dict:
    """Inverse of format_epoch_log_row, used by tests."""
    parts = row.rstrip('\n').split('\t')
    if len(parts) != 6:
        raise ValueError(f'expected 6 tab-separated fields, got {len(parts)}: {row!r}')
    epoch, stage, train_loss, val_loss, val_perf, best = parts
    return {
        'epoch': int(epoch),
        'stage': int(stage),
        'train_loss': float(train_loss),
        'val_loss': float(val_loss),
        'val_perf': float(val_perf),
        'is_best': best == '*',
    }
