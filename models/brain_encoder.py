import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

from utils.utils import (NestedTensor, nested_tensor_from_tensor_list)

from models.backbone import build_backbone
from models.transformer import build_transformer
from models.custom_transformer import build_custom_transformer
from models.vpt import PromptBank, PerROILinearHead, PerVoxelLinearHead

class brain_encoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.lr_backbone = args.lr_backbone

        self.backbone_arch = args.backbone_arch
        self.return_interm = args.return_interm
        self.encoder_arch = args.encoder_arch

        self.lh_vs = args.lh_vs
        self.rh_vs = args.rh_vs

        ### backbone_arch for feature exraction
        self.backbone_model = build_backbone(args)

        # number of brain areas
        self.num_queries = args.num_queries

        #TODO hard  coding the map size for now but fix it
        self.map_size = 31

        ### Brain encoding model
        if 'transformer' in args.encoder_arch:
            if args.encoder_arch == 'transformer':
                self.transformer = build_transformer(args)

            elif self.encoder_arch == 'custom_transformer':
                self.transformer = build_custom_transformer(args)


            self.hidden_dim = self.transformer.d_model
            self.linear_feature_dim  = self.hidden_dim
            self.query_embed = nn.Embedding(self.num_queries, self.hidden_dim)

            if ('resnet' in self.backbone_arch):
                self.input_proj = nn.Conv2d(self.backbone_model.num_channels, self.hidden_dim, kernel_size=1)
        
        elif self.encoder_arch == 'spatial_feature':

            if 'clip' in self.backbone_arch:
                self.map_size = 16

            self.spatial_embed = nn.Embedding(self.num_queries, self.map_size*self.map_size)
            self.linear_feature_dim = self.backbone_model.num_channels

            self.downsize = False
            if self.downsize: 
                self.hidden_dim = 256
                if 'resnet' in self.backbone_arch:
                    stride=1
                    self.map_size = 11
                elif 'clip' in self.backbone_arch:
                    stride=1
                    self.map_size = 8
                else:
                    stride=3
                    self.map_size = 11

                self.input_proj = nn.Conv2d(self.backbone_model.num_channels, self.hidden_dim, kernel_size=3, stride=stride, padding=1)
                    
                # for each roi, learn a spatial map
                self.spatial_embed = nn.Embedding(self.num_queries, self.map_size*self.map_size)
                self.linear_feature_dim = self.hidden_dim

        elif self.encoder_arch == 'vpt':
            # Visual Prompt Tuning: per-ROI soft tokens prepended to a frozen
            # DINOv2 backbone, run once per ROI per image. Two readout modes.
            assert 'dinov2' in self.backbone_arch, 'VPT supports DINOv2 backbones only'
            assert 'cls' not in self.backbone_arch, 'VPT requires the spatial DINOv2 variant'

            self.vpt_readout = args.vpt_readout
            self.vpt_linear_feature = args.vpt_linear_feature
            self.vpt_linear_share = args.vpt_linear_share
            self.vpt_num_prompts_per_roi = args.vpt_num_prompts_per_roi
            self.vpt_roi_chunk = args.vpt_roi_chunk if args.vpt_roi_chunk > 0 else self.num_queries
            self.vpt_prompt_share = getattr(args, 'vpt_prompt_share', 'per_roi')
            self.vpt_decoder_attend_prompts = getattr(args, 'vpt_decoder_attend_prompts', False)
            if self.vpt_decoder_attend_prompts:
                assert self.vpt_prompt_share == 'shared' and args.vpt_readout == 'decoder', \
                    '--vpt_decoder_attend_prompts only applies to shared-prompt + decoder readout'

            D = self.backbone_model.num_channels
            if self.vpt_prompt_share == 'per_roi':
                self.prompt_bank = PromptBank(self.num_queries, self.vpt_num_prompts_per_roi, D)
            elif self.vpt_prompt_share == 'per_voxel':
                # One K-prompt bundle per voxel. self.num_queries == lh_vs + rh_vs
                # for readout_res='voxels'. Backbone runs once per voxel per image
                # (folded into the batch dim, chunked by --vpt_roi_chunk).
                assert args.vpt_readout == 'linear', \
                    '--vpt_prompt_share per_voxel currently requires --vpt_readout linear'
                self.prompt_bank = PromptBank(self.num_queries, self.vpt_num_prompts_per_roi, D)
            else:
                # K shared prompts, one backbone forward per image.
                assert args.vpt_readout == 'decoder', \
                    '--vpt_prompt_share shared currently requires --vpt_readout decoder'
                self.shared_prompts = nn.Parameter(torch.zeros(self.vpt_num_prompts_per_roi, D))
                nn.init.trunc_normal_(self.shared_prompts, std=0.02)

            # The Joiner wraps (dino_module, position_embedding); grab inner refs
            # so we can call forward_with_prompt and rebuild pos_embed for the decoder.
            self.dino = self.backbone_model[0]
            self.pos_embed_module = self.backbone_model[1]

            if self.vpt_readout == 'decoder':
                self.transformer = build_transformer(args)
                self.hidden_dim = self.transformer.d_model
                assert self.hidden_dim == D, \
                    f'VPT decoder readout requires --hidden_dim == backbone dim ({D}); got {self.hidden_dim}'
                if self.vpt_prompt_share == 'per_roi':
                    # ROI identity carried by the prompt; one shared query suffices.
                    self.query_embed = nn.Embedding(1, self.hidden_dim)
                else:
                    # Shared prompts: standard DETR-style queries (one per ROI).
                    self.query_embed = nn.Embedding(self.num_queries, self.hidden_dim)
                self.linear_feature_dim = self.hidden_dim
                self._vpt_skip_universal_heads = False

            elif self.vpt_readout == 'linear':
                if self.vpt_linear_feature == 'prompt_cls_concat':
                    feat_dim = 2 * D
                else:
                    feat_dim = D

                if self.vpt_prompt_share == 'per_voxel':
                    # Per-voxel linear readout: every output unit is a single voxel.
                    # Skip the universal lh_embed/rh_embed (which would map D->vs),
                    # since each voxel only owns a scalar output.
                    self.linear_feature_dim = feat_dim
                    self._vpt_skip_universal_heads = True
                    if self.vpt_linear_share == 'shared':
                        # Single Linear(feat_dim, 1) shared across all voxels;
                        # voxel identity is carried entirely by the prompt.
                        self.shared_voxel_head = nn.Linear(feat_dim, 1)
                    else:
                        # 'per_voxel': one D->1 linear per voxel.
                        self.per_voxel_head = PerVoxelLinearHead(feat_dim, self.num_queries)
                elif self.vpt_linear_share == 'shared':
                    self.linear_feature_dim = feat_dim
                    self._vpt_skip_universal_heads = False
                else:
                    assert hasattr(args, 'lh_roi_masks') and hasattr(args, 'rh_roi_masks'), \
                        'VPT per_roi linear readout requires args.lh_roi_masks / args.rh_roi_masks'
                    self.lh_per_roi_head = PerROILinearHead(feat_dim, args.lh_roi_masks)
                    self.rh_per_roi_head = PerROILinearHead(feat_dim, args.rh_roi_masks)
                    self.n_lh_rois = args.lh_roi_masks.shape[0]
                    self.n_rh_rois = args.rh_roi_masks.shape[0]
                    self.linear_feature_dim = feat_dim
                    self._vpt_skip_universal_heads = True
            else:
                raise ValueError(f'Unknown vpt_readout: {self.vpt_readout}')

        elif self.encoder_arch == 'linear':
            #TODO hard  coding the map size and hidden dimention for now but fix it
            # using conv to make the input smaller for linear layer
            
            if 'resnet' in self.backbone_arch:
                self.hidden_dim = 256
                stride=1
                self.map_size = 11
            elif 'clip' in self.backbone_arch:
                self.hidden_dim = 256
                stride=2
                self.map_size = 8
            else:
                self.hidden_dim = 256
                stride=3
                self.map_size = 11

            #if 'dino' in self.backbone_arch:
            self.input_proj = nn.Conv2d(self.backbone_model.num_channels, self.hidden_dim, kernel_size=3, stride=stride, padding=1)
                
            # if ('resnet' in self.backbone_arch):
            #     self.input_proj = nn.AdaptiveAvgPool2d(1)
            if 'cls' in self.backbone_arch:
                self.hidden_dim = 768
                self.linear_feature_dim  = self.hidden_dim
            else:
                self.linear_feature_dim = self.hidden_dim*self.map_size*self.map_size

        #what is the readout resolution - hemispheres, rois, voxels
        self.readout_res = args.readout_res

        if not getattr(self, '_vpt_skip_universal_heads', False):
            self.lh_embed = nn.Sequential(
                nn.Linear(self.linear_feature_dim, args.lh_vs),
            )

            self.rh_embed = nn.Sequential(
                nn.Linear(self.linear_feature_dim, args.rh_vs),
            )
            

    def forward(self, samples: NestedTensor):

        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        if self.encoder_arch == 'vpt':
            return self._forward_vpt(samples)

        if 'cls' in self.backbone_arch:
            with torch.no_grad():
                input_proj_src = self.backbone_model(samples)

        else:
            if self.lr_backbone == 0:
                with torch.no_grad():
                    features, pos = self.backbone_model(samples)
            else:
                features, pos = self.backbone_model(samples)
        
            input_proj_src, mask = features[-1].decompose()
            assert mask is not None
            pos_embed = pos[-1]
            _,_,h,w = pos_embed.shape


        # print('input_proj_src.shape:', input_proj_src.shape)
        # print('mask.shape:', mask.shape)
        # print(mask)
        # print('pos_embed.shape:', pos_embed.shape)

        # pos_embed = torch.zeros_like(pos_embed).to(pos_embed.device)

        if self.encoder_arch == 'transformer':
            
        # if backbone is resnet, apply 1x1 conv to project the feature to the transformer dimension
            if 'resnet' in self.backbone_arch:
                input_proj_src = self.input_proj(input_proj_src)

            hs = self.transformer(input_proj_src, mask, self.query_embed.weight, pos_embed, self.return_interm)
            output_tokens = hs[-1]

            if self.readout_res == 'voxels':

                lh_f_pred = self.lh_embed(output_tokens[:,0:self.lh_vs,:])
                rh_f_pred = self.rh_embed(output_tokens[:,self.lh_vs:,:])

                lh_f_pred = torch.diagonal(lh_f_pred, dim1=-2, dim2=-1)
                rh_f_pred = torch.diagonal(rh_f_pred, dim1=-2, dim2=-1)

            elif self.readout_res == 'hemis':
                lh_f_pred = self.lh_embed(output_tokens[:,0,:])
                rh_f_pred = self.rh_embed(output_tokens[:,1,:])

            else:
                lh_f_pred = self.lh_embed(output_tokens[:,:output_tokens.shape[1]//2,:])
                lh_f_pred = torch.movedim(lh_f_pred, 1,-1)

                rh_f_pred = self.rh_embed(output_tokens[:,output_tokens.shape[1]//2:,:])
                rh_f_pred = torch.movedim(rh_f_pred, 1,-1)

            out = {'lh_f_pred': lh_f_pred, 'rh_f_pred': rh_f_pred, 'output_tokens': output_tokens}

        elif self.encoder_arch == 'custom_transformer':

            hs = self.transformer(input_proj_src, mask, self.query_embed.weight, pos_embed, self.return_interm)
            output_tokens = hs[-1]

            if self.readout_res == 'voxels':

                lh_f_pred = self.lh_embed(output_tokens[:,0:self.lh_vs,:])
                rh_f_pred = self.rh_embed(output_tokens[:,self.lh_vs:,:])

                lh_f_pred = torch.diagonal(lh_f_pred, dim1=-2, dim2=-1)
                rh_f_pred = torch.diagonal(rh_f_pred, dim1=-2, dim2=-1)

            elif self.readout_res == 'hemis':
                lh_f_pred = self.lh_embed(output_tokens[:,0,:])
                rh_f_pred = self.rh_embed(output_tokens[:,1,:])

            else:
                lh_f_pred = self.lh_embed(output_tokens[:,:output_tokens.shape[1]//2,:])
                lh_f_pred = torch.movedim(lh_f_pred, 1,-1)

                rh_f_pred = self.rh_embed(output_tokens[:,output_tokens.shape[1]//2:,:])
                rh_f_pred = torch.movedim(rh_f_pred, 1,-1)

            out = {'lh_f_pred': lh_f_pred, 'rh_f_pred': rh_f_pred, 'output_tokens': output_tokens}

        elif self.encoder_arch == 'spatial_feature':

            if self.downsize:
                input_proj_src = self.input_proj(input_proj_src)
            
            if self.readout_res == 'rois_all':
                # only for rois_all
                input_proj_src = input_proj_src.flatten(2)
                spatial_map = torch.transpose(self.spatial_embed.weight, 0, 1)
                spatial_map = F.softmax(spatial_map, dim=0)
                output_tokens = torch.matmul(input_proj_src, spatial_map)
                output_tokens = torch.movedim(output_tokens, 1, 2)

                lh_f_pred = self.lh_embed(output_tokens[:,:output_tokens.shape[1]//2,:])
                lh_f_pred = torch.movedim(lh_f_pred, 1,-1)

                rh_f_pred = self.rh_embed(output_tokens[:,output_tokens.shape[1]//2:,:])
                rh_f_pred = torch.movedim(rh_f_pred, 1,-1)

            elif self.readout_res == 'voxels':
                input_proj_src = input_proj_src.flatten(2)
                spatial_map = torch.transpose(self.spatial_embed.weight, 0, 1)
                spatial_map = F.softmax(spatial_map, dim=0)
                output_tokens = torch.matmul(input_proj_src, spatial_map)
                output_tokens = torch.movedim(output_tokens, 1, 2)

                lh_f_pred = self.lh_embed(output_tokens[:,:self.lh_vs,:])
                lh_f_pred = torch.diagonal(lh_f_pred, dim1=-2, dim2=-1)

                rh_f_pred = self.rh_embed(output_tokens[:,self.lh_vs:,:])
                rh_f_pred = torch.diagonal(rh_f_pred, dim1=-2, dim2=-1)


            out = {'lh_f_pred': lh_f_pred, 'rh_f_pred': rh_f_pred, 'output_tokens': output_tokens}

        elif self.encoder_arch == 'linear':
            #if 'dino' in self.backbone_arch:
            if 'cls' not in self.backbone_arch: 
                input_proj_src = self.input_proj(input_proj_src)

            output_tokens = input_proj_src.flatten(1)
            lh_f_pred = self.lh_embed(output_tokens)
            rh_f_pred = self.rh_embed(output_tokens)

            l2_reg = torch.tensor(0.).cuda()
            for param in self.lh_embed.parameters():
                l2_reg += torch.norm(param)

            for param in self.rh_embed.parameters():
                l2_reg += torch.norm(param)  

            out = {'lh_f_pred': lh_f_pred, 'rh_f_pred': rh_f_pred, 'output_tokens': output_tokens, 'l2_reg': l2_reg}


        return out

    def _forward_vpt(self, samples: NestedTensor):
        """VPT forward: per-ROI prompt tokens, one ROI per backbone forward.

        Image batch is replicated across the ROI dim and folded into the batch
        dim, so the backbone runs once with effective batch B*N_ROI. ROIs can
        be processed in chunks of self.vpt_roi_chunk to limit memory.
        """
        if self.vpt_prompt_share == 'shared':
            return self._forward_vpt_shared(samples)

        B = samples.tensors.shape[0]
        H, W = samples.tensors.shape[-2:]
        N_ROI = self.num_queries
        K = self.vpt_num_prompts_per_roi
        D = self.dino.num_channels

        # [B, N_ROI, 3, H, W] then collect chunks along the ROI dim.
        imgs = samples.tensors.unsqueeze(1).expand(B, N_ROI, 3, H, W)
        masks = samples.mask.unsqueeze(1).expand(B, N_ROI, H, W)

        feats_per_roi = []  # entries are dicts of [B*chunk, ...] tensors
        for start in range(0, N_ROI, self.vpt_roi_chunk):
            end = min(start + self.vpt_roi_chunk, N_ROI)
            chunk = end - start

            imgs_c = imgs[:, start:end].reshape(B * chunk, 3, H, W)
            masks_c = masks[:, start:end].reshape(B * chunk, H, W)
            prompts_c = self.prompt_bank.prompts[start:end].unsqueeze(0).expand(B, chunk, K, D).reshape(B * chunk, K, D)

            nested_c = NestedTensor(imgs_c, masks_c)
            out_c = self.dino.forward_with_prompt(nested_c, prompts_c)
            feats_per_roi.append((out_c, chunk))

        # Concat along the ROI dim back into [B, N_ROI, ...] views.
        def gather(field):
            parts = [oc[field].view(B, c, *oc[field].shape[1:]) for oc, c in feats_per_roi]
            return torch.cat(parts, dim=1)

        cls_BR = gather('cls')                    # [B, N_ROI, D]
        prompt_BR = gather('prompt')              # [B, N_ROI, K, D]

        if self.vpt_readout == 'linear':
            if self.vpt_linear_feature == 'prompt':
                feat = prompt_BR.mean(dim=2)  # [B, N_ROI, D]
            elif self.vpt_linear_feature == 'cls':
                feat = cls_BR
            elif self.vpt_linear_feature == 'pooled_patches':
                patches_BR = gather('patches')  # [B, N_ROI, N_patch, D]
                feat = patches_BR.mean(dim=2)
            elif self.vpt_linear_feature == 'prompt_cls_concat':
                feat = torch.cat([prompt_BR.mean(dim=2), cls_BR], dim=-1)
            else:
                raise ValueError(self.vpt_linear_feature)

            if self.vpt_prompt_share == 'per_voxel':
                # feat: [B, N_voxel, feat_dim] with N_voxel == lh_vs + rh_vs.
                if self.vpt_linear_share == 'shared':
                    pred = self.shared_voxel_head(feat).squeeze(-1)   # [B, N_voxel]
                else:
                    pred = self.per_voxel_head(feat)                  # [B, N_voxel]
                lh_f_pred = pred[:, :self.lh_vs]                       # [B, lh_vs]
                rh_f_pred = pred[:, self.lh_vs:]                       # [B, rh_vs]
                return {'lh_f_pred': lh_f_pred, 'rh_f_pred': rh_f_pred,
                        'output_tokens': feat}

            if self.vpt_linear_share == 'shared':
                lh_f_pred = self.lh_embed(feat)        # [B, N_ROI, lh_vs]
                rh_f_pred = self.rh_embed(feat)
                lh_f_pred = torch.movedim(lh_f_pred, 1, -1)  # [B, lh_vs, N_ROI]
                rh_f_pred = torch.movedim(rh_f_pred, 1, -1)
            else:
                feat_lh = feat[:, :self.n_lh_rois]
                feat_rh = feat[:, self.n_lh_rois:]
                lh_f_pred = self.lh_per_roi_head(feat_lh)  # [B, n_lh_rois, lh_vs]
                rh_f_pred = self.rh_per_roi_head(feat_rh)
                # Pad with zeros so the second-axis size matches num_queries (rh half is unused for lh and vice versa).
                lh_pad = lh_f_pred.new_zeros(B, self.n_rh_rois, lh_f_pred.shape[-1])
                rh_pad = rh_f_pred.new_zeros(B, self.n_lh_rois, rh_f_pred.shape[-1])
                lh_f_pred = torch.cat([lh_f_pred, lh_pad], dim=1)
                rh_f_pred = torch.cat([rh_pad, rh_f_pred], dim=1)
                lh_f_pred = torch.movedim(lh_f_pred, 1, -1)
                rh_f_pred = torch.movedim(rh_f_pred, 1, -1)

            return {'lh_f_pred': lh_f_pred, 'rh_f_pred': rh_f_pred,
                    'output_tokens': feat}

        # Decoder readout: cross-attend a single shared query over each ROI's
        # prompt-conditioned patch features. Memory is the patches grid only;
        # pos_embed is recomputed from the per-image mask.
        patches_grid_BR = gather('patches_grid')        # [B, N_ROI, D, h, w]
        mask_small_BR = gather('mask')                  # [B, N_ROI, h, w]

        h, w = patches_grid_BR.shape[-2], patches_grid_BR.shape[-1]
        memory = patches_grid_BR.reshape(B * N_ROI, D, h, w)
        mem_mask = mask_small_BR.reshape(B * N_ROI, h, w)

        pos_embed = self.pos_embed_module(NestedTensor(memory, mem_mask)).to(memory.dtype)
        hs = self.transformer(memory, mem_mask, self.query_embed.weight, pos_embed, self.return_interm)
        # hs: [n_dec_layers, B*N_ROI, num_queries=1, D]
        token = hs[-1][:, 0, :]              # [B*N_ROI, D]
        token = token.view(B, N_ROI, D)

        # Match the existing 'rois_all' convention: lh tokens first half, rh second.
        lh_tokens = token[:, :N_ROI // 2, :]
        rh_tokens = token[:, N_ROI // 2:, :]
        lh_f_pred = torch.movedim(self.lh_embed(lh_tokens), 1, -1)
        rh_f_pred = torch.movedim(self.rh_embed(rh_tokens), 1, -1)

        return {'lh_f_pred': lh_f_pred, 'rh_f_pred': rh_f_pred,
                'output_tokens': token}

    def _forward_vpt_shared(self, samples: NestedTensor):
        """VPT with shared prompts: K learnable tokens prepended once per image,
        single backbone forward, then a standard cross-attn decoder with N_ROI
        queries reads the prompt-conditioned patch grid.
        """
        B = samples.tensors.shape[0]
        K = self.vpt_num_prompts_per_roi
        D = self.dino.num_channels
        N_ROI = self.num_queries

        prompts = self.shared_prompts.unsqueeze(0).expand(B, K, D)
        out_dino = self.dino.forward_with_prompt(samples, prompts)

        memory = out_dino['patches_grid']    # [B, D, h, w]
        mem_mask = out_dino['mask']          # [B, h, w]
        pos_embed = self.pos_embed_module(NestedTensor(memory, mem_mask)).to(memory.dtype)

        if self.vpt_decoder_attend_prompts:
            # Concatenate prompt tokens into the decoder memory so cross-attn
            # sees patches AND prompts. We bypass self.transformer's 4D-input
            # path and drive its encoder/decoder directly on a flat sequence.
            patches_seq = memory.flatten(2).permute(2, 0, 1)               # [Lp, B, D]
            patches_pos = pos_embed.flatten(2).permute(2, 0, 1)            # [Lp, B, D]
            patches_pad = mem_mask.flatten(1)                              # [B, Lp]

            prompt_seq = out_dino['prompt'].permute(1, 0, 2).to(memory.dtype)  # [K, B, D]
            prompt_pos = torch.zeros_like(prompt_seq)                      # no spatial pos for prompts
            prompt_pad = torch.zeros(B, K, dtype=patches_pad.dtype,
                                     device=patches_pad.device)            # never pad prompts

            mem_seq = torch.cat([patches_seq, prompt_seq], dim=0)          # [Lp+K, B, D]
            mem_pos = torch.cat([patches_pos, prompt_pos], dim=0)
            mem_pad = torch.cat([patches_pad, prompt_pad], dim=1)

            tr = self.transformer
            query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)
            tgt = torch.zeros_like(query_embed)
            if tr.num_encoder_layers > 0:
                memory_layers, _ = tr.encoder(mem_seq, src_key_padding_mask=mem_pad, pos=mem_pos)
                enc_out = memory_layers[tr.enc_output_layer]
            else:
                enc_out = mem_seq
            hs = tr.decoder(tgt, enc_out, memory_key_padding_mask=mem_pad,
                            pos=mem_pos, query_pos=query_embed).transpose(1, 2)
        else:
            hs = self.transformer(memory, mem_mask, self.query_embed.weight, pos_embed, self.return_interm)
        output_tokens = hs[-1]               # [B, N_ROI, D]

        lh_tokens = output_tokens[:, :N_ROI // 2, :]
        rh_tokens = output_tokens[:, N_ROI // 2:, :]
        lh_f_pred = torch.movedim(self.lh_embed(lh_tokens), 1, -1)
        rh_f_pred = torch.movedim(self.rh_embed(rh_tokens), 1, -1)

        return {'lh_f_pred': lh_f_pred, 'rh_f_pred': rh_f_pred,
                'output_tokens': output_tokens}
