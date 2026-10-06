"""
VideoMAE encoder: frozen video-native baseline for the backbone comparison.

Why VideoMAE: it is self-supervised video pretraining like V-JEPA 2, at the
same size (ViT-L, ~300M), but it reconstructs masked PIXELS instead of
predicting masked LATENT features, and `MCG-NJU/videomae-large` was
pretrained on Kinetics-400 only (no SSv2). So it separates three questions:
    video-native vs image (CLIP)          — does any video model help?
    latent vs pixel prediction (V-JEPA)   — is it the JEPA objective?
    SSv2 exposure                         — VideoMAE never saw SSv2

Interface matches VisualEncoder (V-JEPA 2), so extract_features.py reuses
the same pooling:
    input  : raw clips (B, T, 3, H, W) in [0, 1]; resized here to 224 and
             ImageNet-normalised. T must equal the checkpoint's num_frames
             (16): VideoMAE's position embeddings are fixed to that grid.
    output : (B, T/2, 196, 1024) — tubelet 2, 14×14 patches per slot.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_HF_MODEL_ID = "MCG-NJU/videomae-large"
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class VideoMAEEncoder(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        cfg = backbone.config
        self.embed_dim = cfg.hidden_size
        self.patch_size = cfg.patch_size
        self.tubelet_size = cfg.tubelet_size
        self.input_resolution = cfg.image_size
        self.num_frames = cfg.num_frames
        self.grid_size = self.input_resolution // self.patch_size
        self.num_patches = self.grid_size ** 2
        self.register_buffer("_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1), persistent=False)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        logger.info("VideoMAEEncoder: %d frames @ %dpx, %d patches/slot, dim %d, frozen",
                    self.num_frames, self.input_resolution, self.num_patches, self.embed_dim)

    @classmethod
    def load_pretrained(cls, model_id: str = _HF_MODEL_ID, device=None,
                        dtype: torch.dtype = torch.float32) -> "VideoMAEEncoder":
        from transformers import VideoMAEModel
        backbone = VideoMAEModel.from_pretrained(model_id, local_files_only=True, dtype=dtype)
        logger.info("VideoMAEEncoder: loaded %s (dtype=%s)", model_id, dtype)
        enc = cls(backbone)
        return enc.to(device) if device is not None else enc

    @classmethod
    def from_config(cls, config) -> "VideoMAEEncoder":
        from transformers import VideoMAEModel
        return cls(VideoMAEModel(config))

    def train(self, mode: bool = True) -> "VideoMAEEncoder":
        super().train(mode)
        self.backbone.eval()
        return self

    @property
    def device(self) -> torch.device:
        return next(self.backbone.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.backbone.parameters()).dtype

    def preprocess_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """(..., 3, H, W) in [0, 1] → resize short side + centre crop + ImageNet norm."""
        lead, (H, W) = frames.shape[:-3], frames.shape[-2:]
        x = frames.reshape(-1, 3, H, W).to(self.device, torch.float32)
        R = self.input_resolution
        if (H, W) != (R, R):
            scale = R / min(H, W)
            nh, nw = max(R, round(H * scale)), max(R, round(W * scale))
            x = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False, antialias=True)
            top, left = (nh - R) // 2, (nw - R) // 2
            x = x[:, :, top:top + R, left:left + R]
        x = (x - self._mean.to(x.device)) / self._std.to(x.device)
        return x.reshape(*lead, 3, R, R)

    @torch.no_grad()
    def forward(self, clips: torch.Tensor, raw: bool = True) -> torch.Tensor:
        if clips.dim() != 5:
            raise ValueError(f"expects (B, T, C, H, W); got {tuple(clips.shape)}")
        if clips.shape[1] != self.num_frames:
            raise ValueError(f"VideoMAE needs exactly {self.num_frames} frames, got {clips.shape[1]}")
        x = self.preprocess_frames(clips) if raw else clips.to(self.device)
        B = x.shape[0]
        tokens = self.backbone(pixel_values=x.to(self.dtype)).last_hidden_state   # (B, T'*P, D)
        T_slots = self.num_frames // self.tubelet_size
        return tokens.view(B, T_slots, self.num_patches, self.embed_dim).float()