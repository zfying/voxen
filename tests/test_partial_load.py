import torch
from torch import nn

from staged import partial_load_state_dict


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone_model = nn.Linear(4, 4)
        self.lh_embed = nn.Sequential(nn.Linear(4, 6))
        self.transformer = nn.Linear(4, 4)


def test_full_match():
    m = Tiny()
    src = {k: torch.randn_like(v) for k, v in m.state_dict().items()
           if not k.startswith('backbone_model')}
    report = partial_load_state_dict(m, src)
    assert set(report['loaded']) == set(src.keys())
    assert report['skipped'] == []


def test_shape_mismatch_skipped():
    m = Tiny()
    src = {'lh_embed.0.weight': torch.zeros(99, 99)}  # wrong shape
    report = partial_load_state_dict(m, src)
    assert report['loaded'] == []
    assert any('shape-mismatch' in reason for _, reason in report['skipped'])


def test_extra_key_skipped():
    m = Tiny()
    src = {'does_not_exist.weight': torch.zeros(2)}
    report = partial_load_state_dict(m, src)
    assert report['loaded'] == []
    assert any(k == 'does_not_exist.weight' for k, _ in report['skipped'])


def test_missing_in_src_recorded():
    m = Tiny()
    src = {'lh_embed.0.weight': torch.zeros(6, 4),
           'lh_embed.0.bias': torch.zeros(6)}
    report = partial_load_state_dict(m, src)
    assert 'transformer.weight' in report['missing_in_src']


def test_backbone_prefix_skipped():
    m = Tiny()
    bb_w = m.state_dict()['backbone_model.weight']
    src = {'backbone_model.weight': torch.zeros_like(bb_w)}
    report = partial_load_state_dict(m, src)
    assert report['loaded'] == []
    assert any(reason == 'prefix-skipped' for _, reason in report['skipped'])
    # And the destination is unchanged.
    assert torch.equal(m.state_dict()['backbone_model.weight'], bb_w)


def test_partial_load_actually_writes_values():
    m = Tiny()
    target = torch.full((6, 4), 7.0)
    src = {'lh_embed.0.weight': target.clone()}
    partial_load_state_dict(m, src)
    assert torch.equal(m.lh_embed[0].weight.data, target)
