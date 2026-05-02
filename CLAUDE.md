# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

Implementation of the paper *Transformer brain encoders explain human high-level visual responses* (Adeli, Sun & Kriegeskorte, 2025; arXiv:2505.17329). Trains models that map images to fMRI responses on the NSD/Algonauts 2023 dataset, using a frozen image backbone (DINOv2, CLIP, or ResNet) followed by a transformer decoder whose learned queries correspond to brain ROIs and route image tokens via cross-attention.

## Common commands

Train a model (canonical example from the README):

```bash
python main.py --run 1 --subj 1 --enc_output_layer 1 --readout_res 'rois_all'
```

Key flags (see `get_args_parser` in `main.py` for full list):
- `--subj {1..8}` — NSD subject id.
- `--backbone_arch {dinov2, dinov2_q, dinov2_cls, dinov2_q_cls, clip, clip_cls, resnet18, resnet50}` — frozen image feature extractor.
- `--encoder_arch {transformer, custom_transformer, spatial_feature, linear, vpt}` — mapping from image features to fMRI. `linear` adds a built-in ridge penalty (`0.02 * l2_reg`). `vpt` enables Visual Prompt Tuning (see below).
- `--readout_res {voxels, rois_all, streams_inc, visuals, bodies, faces, places, words, hemis}` — what the decoder queries correspond to. `rois_all` = one query per ROI; `voxels` = one query per voxel; `hemis` = one per hemisphere. `num_queries` is overridden by `roi_masks(...)` based on this choice — the CLI `--num_queries` is ignored for `nsd_algo`.
- `--enc_output_layer` — which backbone layer feeds the decoder.
- `--saved_feats dinov2q` (with `--saved_feats_dir`) — use precomputed backbone features instead of running the backbone live.
- `--data_dir` — Algonauts 2023 dir; subject suffix (`subj01`, ...) is appended automatically inside `main`.
- `--save_model 1` — checkpoint best-val model (backbone weights are stripped before saving).
- `--wandb_p <project>` — enable W&B logging (defaults to offline mode unless `--wandb_p` is set).

### Visual Prompt Tuning (`--encoder_arch vpt`)

VPT-Shallow variant that prepends per-ROI learnable prompt tokens to a frozen DINOv2 backbone. Backbone runs once per ROI per image (effective batch `B·N_ROI`); user accepted ~38× compute penalty in exchange for clean per-ROI self-attention. Requires `--readout_res rois_all` and a DINOv2 backbone (`dinov2` or `dinov2_q`); incompatible with `--saved_feats`.

VPT-specific flags:
- `--vpt_readout {decoder, linear}` (default `linear`) — decoder uses the existing DETR-style cross-attn with a single shared query; linear is a direct projection.
- `--vpt_linear_feature {prompt, cls, pooled_patches, prompt_cls_concat}` (default `prompt`) — feature fed to linear readout. With K>1 prompts per ROI, `prompt` is mean-pooled.
- `--vpt_linear_share {shared, per_roi, per_voxel}` (default `shared`) — `shared` reuses the universal `lh_embed`/`rh_embed`; `per_roi` builds a `PerROILinearHead` (one Linear per ROI, sized to that ROI's vertex count, scattered into the hemisphere with zeros elsewhere); `per_voxel` builds a `PerVoxelLinearHead` (one D→1 Linear per voxel, applies only with `--vpt_prompt_share per_voxel`).
- `--vpt_num_prompts_per_roi K` (default 1).
- `--vpt_roi_chunk N` (default 0 = all ROIs at once) — split the ROI/voxel dim into chunks of N inside the forward to reduce peak memory. Required (>0) for `--vpt_prompt_share per_voxel`.
- `--vpt_prompt_share {per_roi, shared, per_voxel}` (default `per_roi`) — `per_roi` folds ROI into batch (~N_ROI× compute); `shared` keeps K prompts shared across ROIs with a single backbone forward, requires `--vpt_readout decoder`; `per_voxel` folds voxel into batch (~N_voxel× compute, ~thousands of forwards/image), requires `--readout_res voxels` and `--vpt_readout linear`.
- `--vpt_decoder_attend_prompts` (flag, default off) — shared+decoder only. By default the decoder cross-attends only over the patch grid (prompts influence the decoder *indirectly* via the frozen backbone's self-attention). With this flag the prompt tokens are also concatenated into the decoder memory, so the queries cross-attend over patches **and** prompts directly. Implemented in `_forward_vpt_shared` by bypassing `Transformer.forward` and driving its encoder/decoder on a flat `[Lp+K, B, D]` sequence (zero pos-embed for prompt positions, no padding).

Smoke run: `python main.py --epochs 1 --subj 1 --encoder_arch vpt --vpt_readout linear --vpt_linear_feature prompt --vpt_linear_share shared --readout_res rois_all --backbone_arch dinov2 --batch_size 2 --vpt_roi_chunk 10`.

#### Per-voxel VPT (`--vpt_prompt_share per_voxel`)

Each voxel owns its own K-token prompt bundle, injected into the frozen DINOv2 backbone. Backbone runs once per voxel per image (effective batch `B·N_voxel` with `N_voxel = lh_vs + rh_vs ≈ 30k+30k` per subject), folded into the batch dim and chunked by `--vpt_roi_chunk`. This is **~1000× the per-ROI compute cost**; intended as an experimental ceiling, not a routine training mode. Use small `--batch_size` and small `--vpt_roi_chunk`.

Constraints (asserted in `main.py`): requires `--readout_res voxels`, `--vpt_readout linear`, `--vpt_linear_share {shared, per_voxel}`, `--vpt_roi_chunk > 0`, and a DINOv2 backbone.

Readout heads:
- `--vpt_linear_share shared`: a single `nn.Linear(feat_dim, 1)` shared across all voxels. Voxel identity is carried entirely by the prompt. Tiny readout (~`feat_dim+1` params).
- `--vpt_linear_share per_voxel`: `PerVoxelLinearHead` — one `D→1` linear per voxel, stored as a fused `[N_voxel, D]` weight + `[N_voxel]` bias. Roughly `60k * 768 ≈ 46M` params for D=768, K=1.

The output is `{lh_f_pred: [B, lh_vs], rh_f_pred: [B, rh_vs]}` (LH-first / RH-second voxel ordering, matching `roi_masks('voxels', ...)`); `SetCriterion`'s `readout_res=='voxels'` branch already takes MSE directly on this shape, so no loss-side changes were needed.

Smoke run: `python main.py --epochs 1 --subj 1 --encoder_arch vpt --vpt_prompt_share per_voxel --vpt_readout linear --vpt_linear_feature prompt --vpt_linear_share shared --vpt_num_prompts_per_roi 1 --readout_res voxels --backbone_arch dinov2 --batch_size 1 --vpt_roi_chunk 64`.

### 3-stage VPT (`--vpt_staged`)

Curriculum that splits VPT training into (1) readout only, (2) prompts only, (3) joint. A fresh `AdamW` and a fresh `LinearLR` scheduler are built at each stage transition (LR resets to that stage's full value and decays linearly across the stage). Backbone stays frozen throughout.

Staged-only flags:
- `--vpt_staged` — turns on the 3-stage loop. Overrides `--epochs` with the sum of stage budgets.
- `--vpt_stage{1,2,3}_epochs N` (default 5 each).
- `--vpt_stage{1,2,3}_lr LR` (default = `--lr`) — per-stage LR; lets stage 2 (prompts) use a higher LR than the more delicate joint stage, etc.
- `--vpt_stage_lr_total_iters N` (default = that stage's epoch count) — `LinearLR.total_iters`.
- `--vpt_stage_lr_end_factor F` (default 0.0) — LR at the end of each stage = `stage_lr * F`. 0.0 = decay to zero.
- `--vpt_load_readout {auto, none, PATH}` (default `auto`) — controls stage 1:
  - `auto`: look for a shape-compatible non-VPT baseline checkpoint at `{output_path}/nsd_test/{backbone}_transformer/subj_{subj}/{readout_res}/enc_{enc}/run_{run}/checkpoint.pth`. If found, partial-load matching keys (`lh_embed.*`, `rh_embed.*`, `transformer.*`, `query_embed.*`) and **skip stage 1 entirely**. Resolution lives in `staged.resolve_baseline_checkpoint`.
  - `none`: always train stage 1 in-run for `--vpt_stage1_epochs` epochs.
  - explicit path: same as `auto` but force-load from this file.
- Compatibility: `auto`-load only works for VPT readouts whose readout shape matches the transformer baseline (`--vpt_readout decoder` with backbone hidden dim 768). VPT-linear has no compatible baseline, so stage 1 always runs in-run there.

Staged runs append `-staged` to `arch_tag` so they don't collide with non-staged runs. The whole staged path is built around helpers in `staged.py` (`vpt_param_groups`, `set_stage`, `partial_load_state_dict`, `resolve_baseline_checkpoint`, `build_arch_tag`, `format_epoch_log_row`); these are pure (no I/O, no CUDA) and unit-tested in `tests/`.

Results are written to `{output_path}/nsd_test/{arch_tag}/subj_{subj}/{readout_res}/enc_{enc_output_layer}/run_{run}/`. For non-VPT encoders `arch_tag = {backbone_arch}_{encoder_arch}`; for VPT it is `{backbone_arch}_vpt-{vpt_prompt_share}-{vpt_readout}-{vpt_linear_feature}-{vpt_linear_share}-K{K}` plus a trailing `-attP` when `--vpt_decoder_attend_prompts` is set, and a trailing `-staged` when `--vpt_staged` is set. Each run dir contains `params.txt`, `val_results.txt`, per-vertex correlation arrays, and per-epoch best test predictions. `visualize_results.ipynb` reads these; `visualize_vpt_experiments.ipynb` is the VPT-experiment-specific copy used for the sweeps below.

`val_results.txt` is now a TSV: a header row (`epoch\tstage\ttrain_loss\tval_loss\tval_perf\tbest`) followed by **one row per epoch** (not just on improvement); the `best` column is `*` when the row was a new best-val. wandb (when `--wandb_p` is set) logs `{epoch, stage, train_loss, val_loss, val_perf}` plus per-cluster ROI means every epoch. Stage = 0 for non-staged runs, 1/2/3 for staged.

### VPT experiment scripts (`scripts/vpt/` + `scripts/run_vpt_experiments.sh`)

Subj 1 sweep used to compare VPT variants. Each experiment is its own script under `scripts/vpt/`, sharing config via `scripts/vpt/_common.sh` (sourced):

- `exp1_linear.sh` — baseline linear (`dinov2_q_linear`); passes `--save_model 1`.
- `exp2_transformer.sh` — baseline transformer (`dinov2_q_transformer`); passes `--save_model 1` so exp6 can auto-load its readout.
- `exp3_shared_vpt.sh` — shared VPT + decoder, patches-only memory, K ∈ {1, 5, 10, 20, 40}
- `exp4_per_roi_vpt.sh` — per-ROI VPT + linear, K=1 (~50× backbone compute)
- `exp5_shared_vpt_attend_prompts.sh` — shared VPT + decoder with `--vpt_decoder_attend_prompts`, K ∈ {1, 5, 10, 20, 40}
- `exp6_staged_vpt.sh` — 3-stage shared VPT + decoder; `--vpt_load_readout auto` will skip stage 1 by partial-loading from exp2's checkpoint when present. Per-stage budget via `S1`/`S2`/`S3` env vars (default 5 each); per-stage LR via `LR1`/`LR2`/`LR3` (default `$LR`); same `KS` sweep as exp3.

Each script takes one optional positional arg (`GPU_ID`, default 0; exp4 default 1) and reads env-var overrides from `_common.sh` (`SUBJ`, `RUN`, `EPOCHS`, `BATCH_FAST`, `BATCH_VPT`, `ROI_CHUNK`, `LR`, `WANDB_PROJECT`). exp3/exp5/exp6 also accept `KS="20 40"` to override the K sweep.

`scripts/run_vpt_experiments.sh` is a thin orchestrator that delegates to those files. `[all]` (default) runs exp4 in the background on GPU 1 and exp1/2/3/5/6 serially on GPU 0 (exp6 last, so exp2's checkpoint is on disk by the time exp6's auto-load runs); `[expN]` runs a single one. All experiments use `enc_output_layer=1`, `readout_res=rois_all`, `backbone=dinov2_q`, `lr=5e-4`, `epochs=15` (or per-stage budget for exp6).

`tests/` holds pytest unit tests for `staged.py` helpers (param partitioning, stage freezing, partial state-dict load, baseline-checkpoint resolution, arch_tag construction, log-row format). Run with `pytest tests/ -q` from the repo root. No CUDA or NSD data needed.

## Architecture

**Entry points**
- `main.py` — argparse + training loop. Builds dataloaders, model, criterion; the per-epoch body lives in a closure `run_epochs(...)` that's either called once (legacy) or three times in sequence with fresh per-stage optimizers when `--vpt_staged` is set. Per-vertex correlations and best-val predictions are written inside `run_epochs`; per-epoch metric rows are appended to `val_results.txt` regardless of best-val. Distributed code paths exist but are disabled (see TODO at bottom of `main.py`); always invoked as `main(0, 1, args)`.
- `staged.py` — pure helpers used by the staged path: `vpt_param_groups`, `set_stage`, `partial_load_state_dict`, `resolve_baseline_checkpoint`, `build_arch_tag`, `format_epoch_log_row`. No imports of the model or dataset code, so importing `staged` is cheap.
- `engine.py` — `train_one_epoch`, `evaluate`, `test`, `evaluate_batch`. The `targets` plumbing is unusual: dataloaders return targets as a dict-of-tensors, which is rezipped into a list of per-sample dicts in the engine, then `SetCriterion` indexes `targets[0]` (TODO noted in code).
- `brain_encoder_wrapper.py` — inference-time wrapper that loads one or many trained checkpoints (across runs / encoder layers, sharded across GPUs) and exposes predictions or attention activations. Used by the notebooks.

**Model assembly (`models/`)**
- `brain_encoder.py` — top-level `nn.Module`. Combines `build_backbone(args)` with one of {`build_transformer`, `build_custom_transformer`, spatial-attention map, plain linear}. Two output heads `lh_embed` / `rh_embed` map decoder tokens to per-hemisphere vertices. Output shapes depend on `readout_res`:
  - `voxels`: one token per voxel; predictions read off the diagonal of a `[V, V]` per-hemisphere projection.
  - `hemis`: token 0 → LH, token 1 → RH.
  - everything else (incl. `rois_all`, `streams_inc`): tokens are split in half across hemispheres and projected.
- `backbone.py` dispatches to `dino.py`, `clip.py`, `resnet.py`. Backbones are frozen by default (`lr_backbone=0` triggers `torch.no_grad`); `*_cls` variants return only the CLS token and skip positional embeddings entirely.
- `transformer.py` is a DETR-style encoder–decoder; `custom_transformer.py` is the variant used in the paper (different cross-attention behavior — pick this when reproducing paper numbers if unsure). `position_encoding.py`, `activations.py` round things out.
- `vpt.py` provides `PromptBank` ([N_ROI, K, D] learnable params with per-batch expansion) and `PerROILinearHead` (one Linear per ROI projecting to that ROI's own vertex indices, scattered into the full hemisphere). `dino.py` exposes `forward_with_prompt` on both `dino_model` and `dino_model_with_hooks`, which calls `prepare_tokens_with_masks` then a manual block loop so prompt tokens can be appended after CLS+patches; the qkv hook in `dino_model_with_hooks` captures `seq_len = 1 + N_patch + K`. `brain_encoder._forward_vpt` folds ROI into batch via `unsqueeze(1).expand(...)`, optionally chunks over ROIs, and returns `[B, hemi_vs, N_ROI]` to match `SetCriterion`'s ROI-mask reduction.

**ROI machinery (`datasets/nsd_utils.py`)**
- `roi_maps(data_dir)` loads the Algonauts ROI label files and returns `(roi_name_maps, lh_challenge_rois, rh_challenge_rois)` per ROI family (prf-visualrois, floc-bodies, floc-faces, floc-places, floc-words, streams).
- `roi_masks(readout_res, ...)` returns the per-ROI boolean masks, ROI name lists, and `num_queries` consistent with the chosen `readout_res`. This is the function that determines decoder query count.
- `SetCriterion` (in `main.py`) tiles these ROI masks against per-ROI predictions and sums them into a single per-vertex prediction before MSE — for `streams_inc` / `rois_all` this is how multi-token output collapses to per-vertex loss.

**Data (`datasets/nsd.py`)**
- `algonauts_dataset` for the `nsd_algo` path (Algonauts 2023 challenge layout with `training_split/`, `test_split/`).
- An alternate `nsd_gen` path in `main.py` reads from `/engram/nklab/datasets/natural_scene_dataset/...` and uses a different `nsd_dataset_avg` dataset — note this code path imports `nsd_dataset_avg` without an explicit import in `main.py`, so it currently only works if that symbol is in scope.

## Things to know before editing

- The default config in `main.py` has internal contradictions (e.g. `--objective` defaults to `'classification'` despite only allowing `'NSD'`). Don't infer "intended" defaults from those — read the call sites.
- `args.num_queries` is overwritten inside `main` by `roi_masks(...)` for `nsd_algo`. Changing the CLI flag has no effect there.
- `SetCriterion` references `lh_rois` / `rh_rois` in the non-`streams_inc`/`rois_all` branch but these names aren't defined in scope — that branch is currently broken. Fix or avoid those `readout_res` values.
- Hard-coded paths exist in `brain_encoder_wrapper.py` (`/engram/nklab/hossein/...`) and the `nsd_gen` branch of `main.py`; expect to override these for any non-original environment.
