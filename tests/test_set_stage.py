import pytest

from staged import set_stage, vpt_param_groups


def _grad_state(model):
    return {n: p.requires_grad for n, p in model.named_parameters()}


def test_stage1_readout_only(tiny_vpt_model):
    counts = set_stage(tiny_vpt_model, 1)
    grads = _grad_state(tiny_vpt_model)
    g = vpt_param_groups(tiny_vpt_model)
    for n, _ in g['readout']:
        assert grads[n] is True, n
    for n, _ in g['prompts']:
        assert grads[n] is False, n
    assert counts['prompt_params'] == 0
    assert counts['readout_params'] > 0


def test_stage2_prompts_only(tiny_vpt_model):
    set_stage(tiny_vpt_model, 2)
    grads = _grad_state(tiny_vpt_model)
    g = vpt_param_groups(tiny_vpt_model)
    for n, _ in g['readout']:
        assert grads[n] is False
    for n, _ in g['prompts']:
        assert grads[n] is True


def test_stage3_joint(tiny_vpt_model):
    set_stage(tiny_vpt_model, 3)
    grads = _grad_state(tiny_vpt_model)
    g = vpt_param_groups(tiny_vpt_model)
    for n, _ in g['readout'] + g['prompts']:
        assert grads[n] is True


def test_backbone_always_frozen(tiny_vpt_model):
    for stage in (1, 2, 3):
        set_stage(tiny_vpt_model, stage)
        grads = _grad_state(tiny_vpt_model)
        bb = {n: g for n, g in grads.items() if n.startswith('backbone_model')}
        assert bb and all(g is False for g in bb.values()), (stage, bb)


def test_stage_idempotent_round_trip(tiny_vpt_model):
    set_stage(tiny_vpt_model, 2)
    set_stage(tiny_vpt_model, 1)
    grads = _grad_state(tiny_vpt_model)
    for n, _ in vpt_param_groups(tiny_vpt_model)['prompts']:
        assert grads[n] is False, n


@pytest.mark.parametrize('bad', [0, 4, -1, 'foo'])
def test_invalid_stage_raises(tiny_vpt_model, bad):
    with pytest.raises(ValueError):
        set_stage(tiny_vpt_model, bad)


def test_count_matches_numel(tiny_vpt_model):
    counts = set_stage(tiny_vpt_model, 3)
    expected_readout = sum(p.numel() for _, p in vpt_param_groups(tiny_vpt_model)['readout'])
    expected_prompts = sum(p.numel() for _, p in vpt_param_groups(tiny_vpt_model)['prompts'])
    assert counts['readout_params'] == expected_readout
    assert counts['prompt_params'] == expected_prompts
