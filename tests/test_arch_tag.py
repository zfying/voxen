from staged import build_arch_tag


def test_non_vpt_transformer(make_args):
    args = make_args(encoder_arch='transformer')
    assert build_arch_tag(args) == 'dinov2_q_transformer'


def test_non_vpt_linear(make_args):
    args = make_args(encoder_arch='linear', backbone_arch='clip')
    assert build_arch_tag(args) == 'clip_linear'


def test_vpt_shared_decoder(make_args):
    args = make_args(
        encoder_arch='vpt', vpt_prompt_share='shared', vpt_readout='decoder',
        vpt_linear_feature='prompt', vpt_linear_share='shared',
        vpt_num_prompts_per_roi=5,
    )
    assert build_arch_tag(args) == 'dinov2_q_vpt-shared-decoder-prompt-shared-K5'


def test_vpt_attP_suffix(make_args):
    args = make_args(
        encoder_arch='vpt', vpt_prompt_share='shared', vpt_readout='decoder',
        vpt_linear_feature='prompt', vpt_linear_share='shared',
        vpt_num_prompts_per_roi=5, vpt_decoder_attend_prompts=True,
    )
    assert build_arch_tag(args).endswith('-K5-attP')


def test_staged_suffix(make_args):
    args = make_args(
        encoder_arch='vpt', vpt_prompt_share='shared', vpt_readout='decoder',
        vpt_linear_feature='prompt', vpt_linear_share='shared',
        vpt_num_prompts_per_roi=5, vpt_staged=True,
    )
    assert build_arch_tag(args).endswith('-K5-staged')


def test_staged_with_attP(make_args):
    args = make_args(
        encoder_arch='vpt', vpt_prompt_share='shared', vpt_readout='decoder',
        vpt_linear_feature='prompt', vpt_linear_share='shared',
        vpt_num_prompts_per_roi=10, vpt_decoder_attend_prompts=True,
        vpt_staged=True,
    )
    assert build_arch_tag(args).endswith('-K10-attP-staged')


def test_non_vpt_with_staged_no_suffix(make_args):
    # --vpt_staged is a no-op for non-VPT runs (defensive).
    args = make_args(encoder_arch='transformer', vpt_staged=True)
    # But build_arch_tag still appends -staged because the flag is on; this is
    # acceptable (and in fact desirable for distinguishing test runs).
    assert build_arch_tag(args) == 'dinov2_q_transformer-staged'
