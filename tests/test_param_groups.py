from staged import vpt_param_groups


def _names(group):
    return [n for n, _ in group]


def test_per_roi_partition(tiny_vpt_model):
    g = vpt_param_groups(tiny_vpt_model)
    prompt_names = _names(g['prompts'])
    readout_names = _names(g['readout'])

    assert prompt_names == ['prompt_bank.prompts']
    assert all('backbone_model' not in n for n in readout_names)
    # Readout includes the heads, transformer, and query embedding.
    for needle in ('lh_embed', 'rh_embed', 'transformer', 'query_embed'):
        assert any(needle in n for n in readout_names), needle


def test_shared_prompts_partition(tiny_vpt_model_shared):
    g = vpt_param_groups(tiny_vpt_model_shared)
    assert _names(g['prompts']) == ['shared_prompts']


def test_per_roi_heads_in_readout(tiny_vpt_model_per_roi):
    g = vpt_param_groups(tiny_vpt_model_per_roi)
    readout_names = _names(g['readout'])
    assert any('lh_per_roi_head' in n for n in readout_names)
    assert any('rh_per_roi_head' in n for n in readout_names)


def test_backbone_excluded(tiny_vpt_model):
    g = vpt_param_groups(tiny_vpt_model)
    all_names = _names(g['prompts']) + _names(g['readout'])
    assert not any(n.startswith('backbone_model') for n in all_names)
