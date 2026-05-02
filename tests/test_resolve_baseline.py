import os

from staged import resolve_baseline_checkpoint


def test_auto_finds_existing(make_args, tmp_results_tree):
    args = make_args(
        output_path=tmp_results_tree['output_path'],
        vpt_readout='decoder',
        vpt_load_readout='auto',
    )
    assert resolve_baseline_checkpoint(args) == tmp_results_tree['ckpt_path']


def test_auto_returns_none_when_missing(make_args, tmp_path):
    args = make_args(
        output_path=str(tmp_path) + '/',
        vpt_readout='decoder',
        vpt_load_readout='auto',
    )
    assert resolve_baseline_checkpoint(args) is None


def test_linear_readout_has_no_baseline(make_args, tmp_results_tree):
    # Even when a transformer baseline exists, vpt_readout=linear is incompatible.
    args = make_args(
        output_path=tmp_results_tree['output_path'],
        vpt_readout='linear',
        vpt_load_readout='auto',
    )
    assert resolve_baseline_checkpoint(args) is None


def test_explicit_path_returned_verbatim(make_args, tmp_path):
    explicit = str(tmp_path / 'somewhere.pth')
    args = make_args(vpt_load_readout=explicit, vpt_readout='decoder')
    assert resolve_baseline_checkpoint(args) == explicit


def test_none_returns_none(make_args, tmp_results_tree):
    args = make_args(
        output_path=tmp_results_tree['output_path'],
        vpt_readout='decoder',
        vpt_load_readout='none',
    )
    assert resolve_baseline_checkpoint(args) is None
