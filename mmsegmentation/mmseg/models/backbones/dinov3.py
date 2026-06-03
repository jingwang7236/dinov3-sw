# mmsegmentation/mmseg/models/backbones/dinov3.py
"""
DINOv3 Backbone for MMSegmentation

Supports:
    - ViT-Large (1024 dim, 24 layers)
    - ViT-Giant/7B (1536/2304 dim)
    - Multiple feature level outputs
    - FPN integration
    - Flexible weight loading (pretrained/random)
    - Backbone freezing/unfreezing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import List, Optional, Tuple, Union

from mmengine.logging import print_log
from mmengine.model import BaseModule
from mmseg.registry import MODELS


class Norm2d(nn.Module):
    """LayerNorm on a (N,C,H,W) tensor."""
    
    def __init__(self, c: int, eps: float = 1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(c, eps=eps)

    def forward(self, x):
        # (N,C,H,W) → (N,H,W,C) → LN → back to (N,C,H,W)
        x = self.ln(x.permute(0, 2, 3, 1))
        return x.permute(0, 3, 1, 2).contiguous()


class FPNNeck(nn.Module):
    """Simple FPN neck for multi-scale feature fusion."""
    
    def __init__(self, in_channels: int, out_channels: int, scales: List[float] = [4, 2, 1, 0.5]):
        super().__init__()
        self.scales = scales
        self.ops = nn.ModuleList()
        
        for r in scales:
            if r == 4:
                ops = nn.Sequential(
                    nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2, bias=False),
                    Norm2d(in_channels),
                    nn.GELU(),
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                )
            elif r == 2:
                ops = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            elif r == 1:
                ops = nn.Identity()
            elif r == 0.5:
                ops = nn.MaxPool2d(kernel_size=2, stride=2)
            else:
                raise ValueError(f"Invalid scale factor: {r}")
            self.ops.append(ops)
    
    def forward(self, feats: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        return tuple(op(f) for op, f in zip(self.ops, feats))


@MODELS.register_module()
class DinoV3Backbone(BaseModule):
    """
    DINOv3 Backbone for MMSegmentation.
    
    Supports:
        - ViT-Large (ViT-L/16) - 1024 dim, 24 layers
        - ViT-Giant (ViT-G/16) - 1536 dim, 40 layers
        - ViT-7B (ViT-7B/16) - 2304 dim, 60 layers
    
    Args:
        model_name (str): Model name. Options: 
            'vit_large', 'vit_giant', 'vit_7b',
            'dinov3_vitl16', 'dinov3_vitg16'
        checkpoint_path (str, optional): Path to pretrained checkpoint.
        freeze_backbone (bool): Whether to freeze the backbone parameters.
        out_indices (tuple): Indices of transformer blocks to output features.
            Default: (5, 11, 17, 23) for ViT-L, (9, 19, 29, 39) for ViT-G
        fpn (bool): Whether to use FPN for multi-scale feature fusion.
        fpn_scales (List[float]): Upsampling/downsampling scales for FPN.
        patch_size (int): Patch size (default: 16).
        img_size (int): Expected input image size (default: 512).
        use_grad_checkpoint (bool): Enable gradient checkpointing to save memory.
        init_cfg (dict, optional): Initialization config dict.
        **kwargs: Additional arguments passed to DINOv3 model.
    """
    
    # Model architecture constants
    MODEL_CONFIGS = {
        'vit_large': {
            'embed_dim': 1024,
            'depth': 24,
            'num_heads': 16,
            'mlp_ratio': 4.0,
            'hub_model': 'vit_large_patch16',
        },
        'dinov3_vitl16': {
            'embed_dim': 1024,
            'depth': 24,
            'num_heads': 16,
            'mlp_ratio': 4.0,
            'hub_model': 'dinov3_vitl16',
        },
        'vit_giant': {
            'embed_dim': 1536,
            'depth': 40,
            'num_heads': 24,
            'mlp_ratio': 4.0,
            'hub_model': 'vit_giant_patch16',
        },
        'dinov3_vitg16': {
            'embed_dim': 1536,
            'depth': 40,
            'num_heads': 24,
            'mlp_ratio': 4.0,
            'hub_model': 'dinov3_vitg16',
        },
        'vit_7b': {
            'embed_dim': 2304,
            'depth': 60,
            'num_heads': 36,
            'mlp_ratio': 4.0,
            'hub_model': 'vit_7b_patch16',
        },
    }
    
    # Recommended output indices for different model depths
    RECOMMENDED_OUT_INDICES = {
        24: (5, 11, 17, 23),   # ViT-L: 4 evenly spaced layers
        40: (9, 19, 29, 39),   # ViT-G: 4 evenly spaced layers
        60: (14, 29, 44, 59),  # ViT-7B: 4 evenly spaced layers
    }
    
    def __init__(
        self,
        model_name: str = 'vit_large',
        checkpoint_path: Optional[str] = None,
        freeze_backbone: bool = False,
        out_indices: Optional[Tuple[int, ...]] = None,
        fpn: bool = True,
        fpn_scales: List[float] = [4, 2, 1, 0.5],
        patch_size: int = 16,
        img_size: int = 512,
        use_grad_checkpoint: bool = False,
        init_cfg: Optional[dict] = None,
        **kwargs
    ):
        super().__init__(init_cfg=init_cfg)
        
        # Validate model name
        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(
                f"Unsupported model: {model_name}. "
                f"Available options: {list(self.MODEL_CONFIGS.keys())}"
            )
        
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.model_config = self.MODEL_CONFIGS[model_name]
        self.embed_dim = self.model_config['embed_dim']
        self.depth = self.model_config['depth']
        self.freeze_backbone = freeze_backbone
        self.patch_size = patch_size
        self.img_size = img_size
        self.use_grad_checkpoint = use_grad_checkpoint
        self.fpn = fpn
        
        # Set default out_indices based on model depth if not provided
        if out_indices is None:
            depth = self.depth
            if depth in self.RECOMMENDED_OUT_INDICES:
                self.out_indices = self.RECOMMENDED_OUT_INDICES[depth]
            else:
                # Auto-generate 4 evenly spaced indices
                step = depth // 4
                self.out_indices = tuple(range(step - 1, depth, step))[:4]
        else:
            self.out_indices = tuple(sorted(set(out_indices)))
        
        # Validate out_indices
        for idx in self.out_indices:
            if idx < 0 or idx >= self.depth:
                raise ValueError(
                    f"out_indices {idx} out of range [0, {self.depth-1}]"
                )
        
        # Initialize the backbone
        self._init_backbone(checkpoint_path, **kwargs)
        
        # Setup output channels
        self.out_channels = [self.embed_dim] * len(self.out_indices)
        
        # Setup FPN neck
        if self.fpn:
            self.neck = FPNNeck(
                in_channels=self.embed_dim,
                out_channels=self.embed_dim,
                scales=fpn_scales
            )
        
        # Freeze if requested
        if self.freeze_backbone:
            self._freeze_backbone()
        
        # Print model info
        self._print_info()
    
    def _init_backbone(self, checkpoint_path: Optional[str] = None, **kwargs):
        """Initialize the backbone model."""
        try:
            # Try to load DINOv3 from local path
            # import sys
            # sys.path.insert(0, '/mnt/ht2-nas2/00-model/00-wj/Codes/dinov3-sw/chinasiwei_fm')
            
            # Create backbone based on model type
            # 可选: vit_large, vit_giant, vit_7b
            if self.model_name == "vit_large":
                self.backbone = vit_large(
                    patch_size=self.patch_size,
                    use_grad_checkpoint=self.use_grad_checkpoint,
                    **kwargs
                )
            elif self.model_name == "vit_giant":
                self.backbone = vit_giant(
                    patch_size=self.patch_size,
                    use_grad_checkpoint=self.use_grad_checkpoint,
                    **kwargs
                )
            elif self.model_name == "vit_7b":
                self.backbone = vit_7b(
                    patch_size=self.patch_size,
                    use_grad_checkpoint=self.use_grad_checkpoint,
                    **kwargs
                )
            else:
                # Fallback to generic DinoVisionTransformer
                self.backbone = DinoVisionTransformer(
                    patch_size=self.patch_size,
                    embed_dim=self.embed_dim,
                    depth=self.depth,
                    num_heads=self.model_config['num_heads'],
                    mlp_ratio=self.model_config['mlp_ratio'],
                    use_grad_checkpoint=self.use_grad_checkpoint,
                    **kwargs
                )
            
            # Load checkpoint if provided
            if checkpoint_path:
                self._load_weights(checkpoint_path)
                
        except ImportError as e:
            print_log(f"Failed to import DINOv3: {e}", logger='current', level='ERROR')
            print_log("Please ensure DINOv3 is in your PYTHONPATH", logger='current')
            raise
        except Exception as e:
            print_log(f"Error initializing backbone: {e}", logger='current', level='ERROR')
            raise
    
    def _load_weights(self, checkpoint_path: str):
        """Load pretrained weights from checkpoint."""
        print_log(f"Loading DINOv3 weights from {checkpoint_path}", logger='current')
        
        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            state_dict = checkpoint.get('state_dict', checkpoint)
            
            # Clean up state dict keys
            cleaned_state_dict = {}
            for key, value in state_dict.items():
                # Remove common prefixes
                if key.startswith('backbone.'):
                    new_key = key[9:]
                elif key.startswith('encoder.'):
                    new_key = key[8:]
                elif key.startswith('model.'):
                    new_key = key[6:]
                elif key.startswith('teacher.'):
                    new_key = key[8:]
                elif key.startswith('student.'):
                    new_key = key[8:]
                else:
                    new_key = key
                cleaned_state_dict[new_key] = value
            
            # Load with strict=False to handle missing/unexpected keys
            missing, unexpected = self.backbone.load_state_dict(
                cleaned_state_dict, strict=False
            )
            
            if missing:
                print_log(f"Missing keys (first 10): {missing[:10]}", logger='current')
            if unexpected:
                print_log(f"Unexpected keys (first 10): {unexpected[:10]}", logger='current')
                
            print_log("Successfully loaded DINOv3 weights", logger='current')
            
            # Verify weight loading by checking a few parameters
            sample_param = next(self.backbone.parameters())
            print_log(f"Sample parameter mean: {sample_param.detach().abs().mean():.6f}", 
                     logger='current')
            
        except Exception as e:
            print_log(f"Failed to load checkpoint: {e}", logger='current', level='WARNING')
            print_log("Continuing with random initialization", logger='current')
    
    def _freeze_backbone(self):
        """Freeze all backbone parameters."""
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
        print_log("DINOv3 backbone is frozen", logger='current')
    
    def _print_info(self):
        """Print model information."""
        info = f"""
        ═══════════════════════════════════════════════════════════
        DINOv3 Backbone Configuration
        ═══════════════════════════════════════════════════════════
        Model:           {self.model_name}
        Embed dim:       {self.embed_dim}
        Depth:           {self.depth}
        Patch size:      {self.patch_size}
        Output indices:  {self.out_indices}
        Output levels:   {len(self.out_indices)}
        FPN enabled:     {self.fpn}
        Backbone frozen: {self.freeze_backbone}
        Output channels: {self.out_channels}
        ═══════════════════════════════════════════════════════════
        """
        print_log(info, logger='current')
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.backbone.parameters())
        print_log(f"Backbone parameters: {trainable_params:,} trainable / {total_params:,} total", 
                 logger='current')
    
    def _compute_patch_grid(self, x: torch.Tensor) -> Tuple[int, int, int, int]:
        """Compute patch grid dimensions."""
        B, C, H, W = x.shape
        pH = H // self.patch_size
        pW = W // self.patch_size
        
        # Verify dimensions are compatible
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            print_log(f"Warning: Image size ({H}, {W}) not divisible by patch size ({self.patch_size})",
                     logger='current', level='WARNING')
            
        return B, pH, pW, pH * pW
    
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            List of feature maps from specified layers
        """
        B, pH, pW, num_patches = self._compute_patch_grid(x)
        
        # Get intermediate features (return as sequence, not reshaped)
        try:
            # Try to use get_intermediate_layers with reshape=False
            feats = self.backbone.get_intermediate_layers(
                x,
                n=self.out_indices,
                reshape=False,  # Return sequence format (B, N, D)
                norm=True,
                return_class_token=False,
            )
        except AttributeError:
            # Fallback for backbones without get_intermediate_layers
            feats = self._forward_manual_intermediate(x)
        
        # Reshape each feature map
        reshaped_feats = []
        for feat in feats:
            # feat shape: (B, N, D)
            # Handle potential extra tokens (class token, storage tokens)
            if feat.shape[1] > num_patches:
                # Remove extra tokens (keep only spatial tokens)
                feat = feat[:, :num_patches, :]
            elif feat.shape[1] < num_patches:
                # This shouldn't happen, but handle gracefully
                print_log(f"Warning: Expected {num_patches} patches but got {feat.shape[1]}",
                         logger='current', level='WARNING')
                # Pad if needed
                pad_size = num_patches - feat.shape[1]
                feat = F.pad(feat, (0, 0, 0, pad_size))
            
            # Reshape: (B, D, pH, pW)
            feat = feat.permute(0, 2, 1).reshape(B, -1, pH, pW)
            reshaped_feats.append(feat)
        
        # Convert to float
        feats = [f.float() for f in reshaped_feats]
        
        # Apply FPN if enabled
        if self.fpn:
            feats = self.neck(feats)
            return feats
        
        # Return as tuple for compatibility
        return tuple(feats)
    
    def _forward_manual_intermediate(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Manually extract intermediate features for backbones without get_intermediate_layers."""
        B, pH, pW, _ = self._compute_patch_grid(x)
        
        # Forward through backbone
        x = self.backbone.patch_embed(x)
        
        # Add class token if exists
        if hasattr(self.backbone, 'cls_token'):
            cls_tokens = self.backbone.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embeddings
        if hasattr(self.backbone, 'pos_embed') and self.backbone.pos_embed is not None:
            x = x + self.backbone.pos_embed
        
        # Collect intermediate features
        features = []
        for i, blk in enumerate(self.backbone.blocks):
            x = blk(x)
            if i in self.out_indices:
                # Remove class token and reshape
                feat = x[:, 1:, :] if x.shape[1] > pH * pW else x
                features.append(feat)
        
        return features
    
    def train(self, mode: bool = True):
        """Set train mode."""
        super().train(mode)
        if self.freeze_backbone and hasattr(self, 'backbone'):
            # Ensure backbone stays in eval mode if frozen
            self.backbone.eval()
        return self
    
    def init_weights(self):
        """Initialize weights."""
        super().init_weights()
        
        # If we have init_cfg and it's a checkpoint, load it
        if self.checkpoint_path:
            self._load_weights(self.checkpoint_path)
            print_log("Load params from checkpoint_path: {}".format(self.checkpoint_path), logger='current')
        else:
            # Random initialization
            print_log("Randomly initializing DINOv3 backbone", logger='current')


@MODELS.register_module()
class DinoV3BackboneSimple(DinoV3Backbone):
    """
    Simplified DINOv3 backbone with sensible defaults for segmentation tasks.
    
    This class provides a simplified interface with preset configurations
    optimized for semantic segmentation:
        - Automatically selects appropriate out_indices based on model depth
        - Enables FPN by default for multi-scale feature fusion
        - Uses commonly used fpn_scales
    
    Args:
        model_name (str): Model name (default: 'vit_large')
        checkpoint_path (str, optional): Path to pretrained checkpoint
        freeze_backbone (bool): Whether to freeze backbone (default: False)
        **kwargs: Additional arguments passed to DinoV3Backbone
    """
    
    def __init__(
        self,
        model_name: str = 'vit_large',
        checkpoint_path: Optional[str] = None,
        freeze_backbone: bool = False,
        **kwargs
    ):
        # Set sensible defaults for segmentation
        # out_indices will be auto-selected based on model depth
        # fpn is enabled by default for better multi-scale features
        super().__init__(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            freeze_backbone=freeze_backbone,
            fpn=True,  # Enable FPN by default
            **kwargs
        )


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class RopePositionEmbedding(nn.Module):
    """
    RoPE (Rotary Position Embedding) for Vision Transformers.
    DINOv3 uses RoPE instead of learned position embeddings for better 
    generalization across image resolutions.
    """
    
    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute cosine and sine matrices
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute position indices
        pos = torch.arange(max_seq_len).float()
        freqs = torch.einsum('i,j->ij', pos, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())
    
    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return cos and sin for the given sequence length."""
        return (
            self.cos_cached[:seq_len, :].to(x.device),
            self.sin_cached[:seq_len, :].to(x.device)
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply RoPE to the input tensor."""
    # x shape: (batch, seq_len, num_heads, head_dim)
    x_rotated = x * cos.unsqueeze(0).unsqueeze(0) + rotate_half(x) * sin.unsqueeze(0).unsqueeze(0)
    return x_rotated


class PatchEmbed(nn.Module):
    """Patch Embedding module."""
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class LayerScale(nn.Module):
    """LayerScale module."""
    
    def __init__(self, dim: int, init_values: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class SelfAttention(nn.Module):
    """Self-attention module with optional RoPE."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 12,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_rope: bool = True,
        qk_norm: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_rope = use_rope
        self.qk_norm = qk_norm
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        
        if qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
    
    def forward(
        self,
        x: torch.Tensor,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, C = x.shape
        
        # QKV projection
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, num_heads, N, head_dim)
        
        # Apply RoPE if provided
        if self.use_rope and cos is not None and sin is not None:
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
        
        # Apply QK normalization
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """MLP module."""
    
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SelfAttentionBlock(nn.Module):
    """Transformer block with self-attention."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        layer_scale_init: float = 1e-5,
        use_rope: bool = True,
        qk_norm: bool = True,
    ):
        super().__init__()
        
        self.norm1 = norm_layer(dim)
        self.attn = SelfAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            use_rope=use_rope,
            qk_norm=qk_norm,
        )
        self.ls1 = LayerScale(dim, init_values=layer_scale_init)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=layer_scale_init)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
    
    def forward(
        self,
        x: torch.Tensor,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention block
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), cos=cos, sin=sin)))
        # MLP block
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class DinoVisionTransformer(nn.Module):
    """
    DINOv3 Vision Transformer.
    
    Supports:
        - ViT-Small (21M)
        - ViT-Base (86M) 
        - ViT-Large (307M)
        - ViT-Giant (1B)
        - ViT-7B (7B)
    
    Args:
        img_size: Input image size
        patch_size: Patch size (default: 16)
        in_chans: Number of input channels
        embed_dim: Embedding dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dimension ratio
        qkv_bias: Enable bias for QKV projection
        proj_bias: Enable bias for output projection
        drop_rate: Dropout rate
        attn_drop_rate: Attention dropout rate
        drop_path_rate: Stochastic depth rate
        norm_layer: Normalization layer
        act_layer: Activation function
        layer_scale_init: LayerScale initialization value
        use_rope: Use RoPE positional embeddings
        qk_norm: Normalize Q and K vectors
        use_grad_checkpoint: Enable gradient checkpointing
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        act_layer: nn.Module = nn.GELU,
        layer_scale_init: float = 1e-5,
        use_rope: bool = True,
        qk_norm: bool = True,
        use_grad_checkpoint: bool = False,
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.use_grad_checkpoint = use_grad_checkpoint
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer,
        )
        num_patches = self.patch_embed.num_patches
        
        # RoPE or learnable position embeddings
        self.use_rope = use_rope
        if use_rope:
            self.rope = RopePositionEmbedding(dim=embed_dim // num_heads)
            self.pos_embed = None
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Stochastic depth decay
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            SelfAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                attn_drop=attn_drop_rate,
                proj_drop=drop_rate,
                drop_path=dpr[i],
                act_layer=act_layer,
                norm_layer=norm_layer,
                layer_scale_init=layer_scale_init,
                use_rope=use_rope,
                qk_norm=qk_norm,
            )
            for i in range(depth)
        ])
        
        self.norm = norm_layer(embed_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, N, D)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embeddings (if not using RoPE)
        if not self.use_rope and self.pos_embed is not None:
            x = x + self.pos_embed
        
        # Prepare RoPE cos/sin
        if self.use_rope:
            cos, sin = self.rope(x, seq_len=x.shape[1])
        else:
            cos, sin = None, None
        
        # Pass through transformer blocks
        for blk in self.blocks:
            if self.use_grad_checkpoint and self.training:
                x = checkpoint(blk, x, cos, sin)
            else:
                x = blk(x, cos=cos, sin=sin)
        
        x = self.norm(x)
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        return x
    
    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, List[int]] = 12,
        reshape: bool = False,
        norm: bool = True,
        return_class_token: bool = False,
    ) -> List[torch.Tensor]:
        """
        Extract intermediate layer features.
        
        Args:
            x: Input tensor
            n: Layer indices to extract
            reshape: Reshape to 2D spatial format
            norm: Apply final normalization
            return_class_token: Whether to return class token
        
        Returns:
            List of feature tensors
        """
        if isinstance(n, int):
            n = [n]
        
        B = x.shape[0]
        H, W = self.img_size // self.patch_size, self.img_size // self.patch_size
        
        # Forward patch embedding
        x = self.patch_embed(x)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embeddings (if not using RoPE)
        if not self.use_rope and self.pos_embed is not None:
            x = x + self.pos_embed
        
        # Prepare RoPE cos/sin
        if self.use_rope:
            cos, sin = self.rope(x, seq_len=x.shape[1])
        else:
            cos, sin = None, None
        
        # Collect intermediate features
        features = []
        for i, blk in enumerate(self.blocks):
            x = blk(x, cos=cos, sin=sin)
            if i in n:
                feat = x.clone()
                if norm:
                    feat = self.norm(feat)
                if not return_class_token:
                    feat = feat[:, 1:, :]  # Remove class token
                if reshape:
                    feat = feat.permute(0, 2, 1).reshape(B, -1, H, W)
                features.append(feat)
        
        return features


# Model factory functions

def vit_small(patch_size: int = 16, **kwargs):
    """ViT-Small (21M parameters)."""
    return DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        **kwargs
    )


def vit_base(patch_size: int = 16, **kwargs):
    """ViT-Base (86M parameters)."""
    return DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        **kwargs
    )


def vit_large(patch_size: int = 16, **kwargs):
    """ViT-Large (307M parameters)."""
    return DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        **kwargs
    )


def vit_giant(patch_size: int = 16, **kwargs):
    """ViT-Giant (1B parameters)."""
    return DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        mlp_ratio=4.0,
        **kwargs
    )


def vit_7b(patch_size: int = 16, **kwargs):
    """ViT-7B (7B parameters)."""
    return DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=2304,
        depth=60,
        num_heads=36,
        mlp_ratio=4.0,
        **kwargs
    )