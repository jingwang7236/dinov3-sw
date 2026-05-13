from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.transformer import FFN, PatchEmbed
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import trunc_normal_
from mmengine.runner.checkpoint import _load_checkpoint

from mmpretrain.registry import MODELS

class Norm2d(nn.Module):
    """LayerNorm on a (N,C,H,W) tensor."""
    def __init__(self, c: int):
        super().__init__()
        self.ln = nn.LayerNorm(c, eps=1e-6)

    def forward(self, x):
        # (N,C,H,W) → (N,H,W,C) → LN → back to (N,C,H,W)
        x = self.ln(x.permute(0, 2, 3, 1))
        return x.permute(0, 3, 1, 2).contiguous()


@MODELS.register_module()
class DinoV3Backbone(BaseModule):
    
    def __init__(
        self,
        # model_name='dinov3_vitl16',
        model_name='dinov3_vitl16_chinasiwei',
        checkpoint_path=None,
        # checkpoint_path='/mnt/mty/open_source_mm/chinasiwei_fm/dinov3-main/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth',
        freeze_backbone=True,
        # out_indices=(5, 11, 17, 23), # 
        out_indices=(7, 11, 15, 23),
        fpn=False,
        n_storage_tokens=0,
        mask_k_bias=False,
        untie_global_and_local_cls_norm=False,
        channel_adaptive=False,
        feat_fuse_method='mean'
    ):
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_path) if checkpoint_path else None
        super().__init__(init_cfg=init_cfg)
        REPO_DIR = '/mnt/mty/open_source_mm/chinasiwei_fm/dinov3-main'

        # self.encoder = torch.hub.load(REPO_DIR, model=model_name, source='local', pretrained=False)
        self.encoder = torch.hub.load(
            REPO_DIR, 
            model=model_name, 
            source='local', 
            weights=checkpoint_path,
            n_storage_tokens=n_storage_tokens,
            mask_k_bias=mask_k_bias,
            untie_global_and_local_cls_norm=untie_global_and_local_cls_norm,
            channel_adaptive=channel_adaptive,
            feat_fuse_method=feat_fuse_method
        )
        self.freeze_backbone = freeze_backbone
        # self.encoder.eval()
        if freeze_backbone:
            self._freeze_backbone()

        self.out_indices  = tuple(out_indices)
        self.embed_dim = getattr(self.encoder, 'embed_dim', 1024)
        self.out_channels = [self.embed_dim] * len(self.out_indices)

        self.fpn = fpn
        # self.fp16 = fp16
        rescales = [4, 2, 1, 0.5]
        
        if self.fpn:
            self.ops = nn.ModuleList()
            for r in rescales:
                branch = []
                if r == 4:
                    branch.append(nn.Sequential(
                        nn.ConvTranspose2d(self.embed_dim // 1, self.embed_dim // 1, kernel_size=2, stride=2, bias=False),
                        Norm2d(self.embed_dim // 1),
                        nn.GELU(),
                        nn.ConvTranspose2d(self.embed_dim, self.embed_dim, kernel_size=2, stride=2),
                    ))
                elif r == 2:
                    branch.append(nn.ConvTranspose2d(self.embed_dim // 1, self.embed_dim // 1, kernel_size=2, stride=2))
                elif r == 1:
                    branch.append(nn.Identity())
                elif r == 0.5:
                    branch.append(nn.MaxPool2d(kernel_size=2, stride=2))
                else:
                    raise KeyError(f"Invalid rescale factor: {r}")
                self.ops.append(nn.Sequential(*branch))
        
    def _freeze_backbone(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
    

    def init_weights(self):
        """Load ViT from init_cfg checkpoint, then init only scale_ops."""
        # 1) fingerprint
        wt_sum = sum(p.detach().abs().mean() for p in self.encoder.parameters())
        print(f'\n[DINOv3] pre-load weight abs-mean fingerprint: {wt_sum:.3f}')

        # 2) load checkpoint (MMEngine already passed us the path via init_cfg)
        print(f"Loading from {self.init_cfg['checkpoint']}")
        ckpt = _load_checkpoint(self.init_cfg['checkpoint'], map_location='cpu')
        sd   = ckpt.get('state_dict', ckpt)
        missing, unexpected = self.encoder.load_state_dict(sd, strict=False)
        print(f"[DINOv3] Loaded ViT: missing_keys={missing}, unexpected_keys={unexpected}")

        wt_sum = sum(p.detach().abs().mean() for p in self.encoder.parameters())
        print(f'[DINOv3] post-load weight abs-mean fingerprint: {wt_sum:.3f}')
        
        # 3) (re)initialize only the up/down-sampling heads
        if self.fpn:
            for m in self.ops.modules():
                if isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, Norm2d):
                    nn.init.ones_(m.ln.weight)
                    nn.init.zeros_(m.ln.bias)
                elif isinstance(m, nn.ConvTranspose2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)


    def forward(self, x):
        '''
        if self.training:
            print("=== ViT parameter requires_grad states ===")
            for name, p in self.encoder.named_parameters():
                print(f"{name:60s}  requires_grad={p.requires_grad}")
            print("==========================================")
  
        '''

            
        # with autocast(enabled=True, dtype=torch.float16):
        feats = self.encoder.get_intermediate_layers(
            x,
            # n=[5, 11, 17, 23],
            n=self.out_indices,
            reshape=True,
            norm=True,           
            return_class_token=False,
        )
        feats = [f.float() for f in feats]
        if self.fpn:
            return tuple(op(f) for op,f in zip(self.ops, feats))
        else:
            return feats