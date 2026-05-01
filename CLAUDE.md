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
- `--vpt_linear_share {shared, per_roi}` (default `shared`) — `shared` reuses the universal `lh_embed`/`rh_embed`; `per_roi` builds a `PerROILinearHead` (one Linear per ROI, sized to that ROI's vertex count, scattered into the hemisphere with zeros elsewhere).
- `--vpt_num_prompts_per_roi K` (default 1).
- `--vpt_roi_chunk N` (default 0 = all ROIs at once) — split the ROI dim into chunks of N inside the forward to reduce peak memory.

Smoke run: `python main.py --epochs 1 --subj 1 --encoder_arch vpt --vpt_readout linear --vpt_linear_feature prompt --vpt_linear_share shared --readout_res rois_all --backbone_arch dinov2 --batch_size 2 --vpt_roi_chunk 10`.

Results are written to `{output_path}/nsd_test/{backbone_arch}_{encoder_arch}/subj_{subj}/{readout_res}/enc_{enc_output_layer}/run_{run}/` and include `params.txt`, `val_results.txt`, per-vertex correlation arrays, and per-epoch best test predictions. `visualize_results.ipynb` reads these.

There is no test suite, lint config, or build step.

## Architecture

**Entry points**
- `main.py` — argparse + training loop. Builds dataloaders, model, criterion; runs `train_one_epoch` / `evaluate` / `test` per epoch; writes predictions and correlations whenever validation improves. Distributed code paths exist but are disabled (see TODO at bottom of `main.py`); always invoked as `main(0, 1, args)`.
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
