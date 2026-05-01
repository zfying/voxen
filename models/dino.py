# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DINO Backbone modules.
"""

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List, Callable

from utils.utils import NestedTensor

from .position_encoding import build_position_encoding
    
    
class dino_model_with_hooks(nn.Module):

    def __init__(self, enc_output_layer, return_interm_layers= False, return_cls=False):
        super().__init__()   
        
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.num_channels = 768
        
        for name, parameter in self.backbone.named_parameters():
            parameter.requires_grad_(False)
            
        self.qkv_feats = {'qkv_feats':torch.empty(0)}

        self.enc_output_layer = enc_output_layer
        self.backbone._modules["blocks"][enc_output_layer]._modules["attn"]._modules["qkv"].register_forward_hook(self.hook_fn_forward_qkv)  #self.hook_fn_forward_qkv())

        self.return_interm_layers = return_interm_layers
        self.return_cls = return_cls

    def hook_fn_forward_qkv(self, module, input, output) -> Callable:
#         def fn(_, __, output):
        self.qkv_feats['qkv_feats'] = output
            
            
    def forward(self, tensor_list: NestedTensor):
        xs = tensor_list.tensors

        #print(xs.shape)
        h, w = int(xs.shape[2]/14), int(xs.shape[3]/14)

#         self.qkv_feats = []
#         qkv_feats = []

#         self.backbone._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(lambda self, input, output: qkv_feats.append(output))

        xs = self.backbone.get_intermediate_layers(xs)[0]

        feats = self.qkv_feats['qkv_feats']
        # Dimensions
        nh = 12 #Number of heads

        feats = feats.reshape(xs.shape[0], xs.shape[1]+1, 3, nh, -1 // nh).permute(2, 0, 3, 1, 4)
        q, k, v = feats[0], feats[1], feats[2]
        q = q.transpose(1, 2).reshape(xs.shape[0], xs.shape[1]+1, -1)

        xs = q[:,1:,:]

        if self.return_cls:
            #out['cls_token'] = q[:,0:1,:]
            return q[:,0,:]

        xs = {'layer_top':xs}
#         xs = self.body(tensor_list.tensors)

        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None

            x = torch.reshape(x, (x.shape[0],h,w,self.num_channels)).permute(0,3,1,2)

            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out

    def forward_with_prompt(self, tensor_list: NestedTensor, prompt: torch.Tensor):
        """Run DINOv2 with `prompt` ([B, K, D]) appended after CLS+patch tokens.

        Returns a dict with keys cls [B, D], patches [B, N_patch, D] (post-norm
        at enc_output_layer), patches_grid [B, D, h, w], prompt [B, K, D],
        q_prompt [B, K, D] (post-qkv-Q at the hook layer, prompt positions),
        and mask [B, h, w] interpolated from the input mask.
        """
        xs = tensor_list.tensors
        h_p = int(xs.shape[2] / 14)
        w_p = int(xs.shape[3] / 14)

        x = self.backbone.prepare_tokens_with_masks(xs, masks=None)  # [B, 1+N, D]
        n_patch = x.shape[1] - 1
        x = torch.cat([x, prompt], dim=1)  # [B, 1+N+K, D]

        layer_outputs = []
        for blk in self.backbone.blocks:
            x = blk(x)
            layer_outputs.append(x)

        # Pick output of the requested layer; apply final norm if it's the last.
        idx = self.enc_output_layer
        if idx < 0:
            idx = len(self.backbone.blocks) + idx
        out_x = layer_outputs[idx]
        if idx == len(self.backbone.blocks) - 1:
            out_x = self.backbone.norm(out_x)

        cls = out_x[:, 0]
        patches = out_x[:, 1:1 + n_patch]
        prompt_out = out_x[:, 1 + n_patch:]

        # q-features captured by the hook (registered at -enc_output_layer in
        # __init__, but here enc_output_layer is positive layer index): pull
        # q from the qkv output and split off the prompt-position rows.
        feats = self.qkv_feats['qkv_feats']
        nh = 12
        feats = feats.reshape(x.shape[0], x.shape[1], 3, nh, -1 // nh).permute(2, 0, 3, 1, 4)
        q = feats[0].transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)
        q_prompt = q[:, 1 + n_patch:, :]

        patches_grid = patches.transpose(1, 2).reshape(patches.shape[0], self.num_channels, h_p, w_p)
        m = tensor_list.mask
        assert m is not None
        mask_small = F.interpolate(m[None].float(), size=(h_p, w_p)).to(torch.bool)[0]

        return {
            'cls': cls,
            'patches': patches,
            'patches_grid': patches_grid,
            'prompt': prompt_out,
            'q_prompt': q_prompt,
            'mask': mask_small,
        }


class dino_model(nn.Module):

    def __init__(self, enc_output_layer, return_interm_layers= False, return_cls=False):
        super().__init__()   
        
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.num_channels = 768
        
        for name, parameter in self.backbone.named_parameters():
            parameter.requires_grad_(False)
            
        self.enc_output_layer = enc_output_layer
        self.return_interm_layers = return_interm_layers
        self.return_cls = return_cls
        

    def forward(self, tensor_list: NestedTensor):
        xs = tensor_list.tensors
        
        patch_size = 14
    
        w_p = int(xs.shape[2] / patch_size)
        h_p = int(xs.shape[3] / patch_size)
        
        xs = self.backbone.get_intermediate_layers(xs, n=12) #[0]

        if self.return_interm_layers:
            xs = {'0':xs[0], '1':xs[1], '2':xs[2], '3':xs[3], '4':xs[4], '5':xs[5], '6':xs[6], '7':xs[7], '8':xs[8], '9':xs[9], '10':xs[10], '11':xs[11]}
        else:
            xs = {'layer_top':xs[self.enc_output_layer]}
            cls_token = xs[self.enc_output_layer][:,0,:]

        # TODO fix this
        if self.return_cls:
            return cls_token

        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None

            x = torch.reshape(x, (x.shape[0], w_p,h_p,self.num_channels)).permute(0,3,1,2)

            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out

    def forward_with_prompt(self, tensor_list: NestedTensor, prompt: torch.Tensor):
        """Run DINOv2 with `prompt` ([B, K, D]) appended after CLS+patch tokens.

        Same return shape as dino_model_with_hooks.forward_with_prompt, except
        q_prompt is None (no qkv hook in this variant).
        """
        xs = tensor_list.tensors
        h_p = int(xs.shape[2] / 14)
        w_p = int(xs.shape[3] / 14)

        x = self.backbone.prepare_tokens_with_masks(xs, masks=None)  # [B, 1+N, D]
        n_patch = x.shape[1] - 1
        x = torch.cat([x, prompt], dim=1)  # [B, 1+N+K, D]

        layer_outputs = []
        for blk in self.backbone.blocks:
            x = blk(x)
            layer_outputs.append(x)

        idx = self.enc_output_layer
        if idx < 0:
            idx = len(self.backbone.blocks) + idx
        out_x = layer_outputs[idx]
        if idx == len(self.backbone.blocks) - 1:
            out_x = self.backbone.norm(out_x)

        cls = out_x[:, 0]
        patches = out_x[:, 1:1 + n_patch]
        prompt_out = out_x[:, 1 + n_patch:]

        patches_grid = patches.transpose(1, 2).reshape(patches.shape[0], self.num_channels, h_p, w_p)
        m = tensor_list.mask
        assert m is not None
        mask_small = F.interpolate(m[None].float(), size=(h_p, w_p)).to(torch.bool)[0]

        return {
            'cls': cls,
            'patches': patches,
            'patches_grid': patches_grid,
            'prompt': prompt_out,
            'q_prompt': None,
            'mask': mask_small,
        }

