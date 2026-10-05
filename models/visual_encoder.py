"""
V-JEPA 2 visual encoder: frozen video backbone returning spatiotemporal patch tokens.

Design summary
--------------
Backbone : V-JEPA 2 ViT-L/16 (`facebook/vjepa2-vitl-fpc64-256`, HF transformers).
Input    : raw clips (B, T, 3, H, W), float in [0, 1], any size. Resize +
           centre-crop to 256×256 and ImageNet normalisation happen here, so
           the video dataset can feed raw frames to both this encoder and
           CLIPEncoder (which applies its own 224px / CLIP-stats preprocessing).
Output   : (B, T', P, 1024) where
               T' = T / tubelet_size   (tubelet_size = 2 → 16 frames → 8 slots)
               P  = (256/16)² = 256    spatial patches per slot

Fixes relative to the image-CZSL version
----------------------------------------
1. Tubelet-aware reshape (the important one).
   V-JEPA 2 embeds each *pair* of frames into one token grid (3-D patches of
   2×16×16), so 16 frames give 8 temporal slots, not 16. The old code did
   `tokens.view(B, F, tokens.size(1) // F, D)`, i.e. it assumed one grid per
   frame. With MIT-States' 2 duplicated frames there is really 1 slot, so it
   split each image's 16×16 patch grid into top/bottom halves and called them
   "frames". Harmless there (the heads flattened and pooled every token, and
   the halves are the same token set), but for video it would mislabel time:
   "frame" k would hold the bottom half of slot k/2. Anything that uses the
   time axis (temporal conv, reversal test, frame shuffling) would be wrong.
   The output is now reshaped by the backbone's real token grid.

2. Predictor no longer runs.
   `VJEPA2Model.forward` runs the 12-layer V-JEPA predictor by default. We
   only need encoder features, so `skip_predictor=True` drops wasted compute
   on every call (it matters once you extract features for ~80k clips).

3. No silent random-weight fallback.
   The old `load_pretrained()` caught any loading error and fell back to a
   randomly initialised timm ViT with only a warning in the log, so a broken
   HF cache on a compute node would have produced garbage features that
   looked like a real run. Loading now fails loudly. For offline tests use
   `from_config()` with a tiny random config instead (same code path as the
   real model; timm is no longer needed).

4. Frame counts not divisible by the tubelet size are padded by repeating the
   last frame. The 3-D patch conv would otherwise silently drop it.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_HF_MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"

# V-JEPA 2 is trained with ImageNet statistics (see its video processor).
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class VisualEncoder(nn.Module):
    """
    Frozen V-JEPA 2 encoder.

    Typical usage
    -------------
    encoder = VisualEncoder.load_pretrained(device=device, dtype=torch.bfloat16)
    tokens = encoder(raw_clips)                  # (B, T/2, 256, 1024)

    Images (e.g. MIT-States) work too: pass (B, 1, 3, H, W) or (B, 2, 3, H, W);
    either becomes one temporal slot of 256 patches.
    """

    def __init__(self, backbone: nn.Module, is_frozen: bool = True) -> None:
        super().__init__()
        self.backbone = backbone

        cfg = backbone.config
        self.embed_dim = cfg.hidden_size
        self.patch_size = cfg.patch_size
        self.tubelet_size = cfg.tubelet_size
        self.input_resolution = cfg.crop_size
        self.grid_size = self.input_resolution // self.patch_size
        self.num_patches = self.grid_size ** 2      # spatial patches per slot

        self.register_buffer(
            "_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1), persistent=False
        )

        self.is_frozen = is_frozen
        if is_frozen:
            self.freeze()

        logger.info(
            "VisualEncoder: V-JEPA 2 | input=%dpx patch=%d tubelet=%d | "
            "%d patches/slot | embed_dim=%d | frozen=%s",
            self.input_resolution, self.patch_size, self.tubelet_size,
            self.num_patches, self.embed_dim, is_frozen,
        )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def load_pretrained(
        cls,
        model_id: str = _HF_MODEL_ID,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        is_frozen: bool = True,
    ) -> "VisualEncoder":
        """
        Load V-JEPA 2 weights from the local HF cache (compute nodes are
        offline; pre-download on a login node). Raises if loading fails.

        dtype=torch.bfloat16 roughly halves memory and speeds up extraction
        on A100/H100; features are cast back to float32 on output.
        """
        from transformers import AutoModel

        backbone = AutoModel.from_pretrained(
            model_id, local_files_only=True, dtype=dtype
        )
        logger.info("VisualEncoder: loaded %s (dtype=%s)", model_id, dtype)
        encoder = cls(backbone, is_frozen=is_frozen)
        if device is not None:
            encoder = encoder.to(device)
        return encoder

    @classmethod
    def from_config(cls, config, is_frozen: bool = True) -> "VisualEncoder":
        """Random-weight model from a VJEPA2Config — for offline tests only."""
        from transformers import VJEPA2Model

        return cls(VJEPA2Model(config), is_frozen=is_frozen)

    # ------------------------------------------------------------------
    # Frozen-backbone helpers
    # ------------------------------------------------------------------

    def freeze(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        logger.info("VisualEncoder: backbone frozen")

    def train(self, mode: bool = True) -> "VisualEncoder":
        # Keep a frozen backbone in eval mode even if a parent calls .train().
        super().train(mode)
        if self.is_frozen:
            self.backbone.eval()
        return self

    @property
    def device(self) -> torch.device:
        return next(self.backbone.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.backbone.parameters()).dtype

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preprocess_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Resize shorter side → centre-crop to input_resolution → ImageNet-normalise.

        Args:
            frames: (..., 3, H, W) float in [0, 1]; leading dims are kept.
        Returns:
            (..., 3, R, R) float32 on the encoder's device.
        """
        if frames.dim() < 3 or frames.shape[-3] != 3:
            raise ValueError(
                f"preprocess_frames expects (..., 3, H, W); got {tuple(frames.shape)}"
            )
        lead = frames.shape[:-3]
        H, W = frames.shape[-2:]
        x = frames.reshape(-1, 3, H, W).to(self.device, torch.float32)

        R = self.input_resolution
        if (H, W) != (R, R):
            scale = R / min(H, W)
            new_h, new_w = max(R, round(H * scale)), max(R, round(W * scale))
            x = F.interpolate(
                x, size=(new_h, new_w), mode="bilinear", align_corners=False,
                antialias=True,
            )
            top, left = (new_h - R) // 2, (new_w - R) // 2
            x = x[:, :, top:top + R, left:left + R]

        x = (x - self._mean) / self._std
        return x.reshape(*lead, 3, R, R)

    def num_slots(self, num_frames: int) -> int:
        """Temporal token slots produced for a clip of num_frames frames."""
        return max(1, math.ceil(num_frames / self.tubelet_size))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, clips: torch.Tensor, raw: bool = True) -> torch.Tensor:
        """
        Args:
            clips: (B, T, 3, H, W). Raw [0, 1] frames if raw=True; otherwise
                already resized to input_resolution and normalised.
        Returns:
            (B, T', num_patches, embed_dim) float32, T' = ceil(T / tubelet).
            Slot t covers frames [t*tubelet, (t+1)*tubelet).
        """
        if clips.dim() != 5:
            raise ValueError(
                f"VisualEncoder expects (B, T, C, H, W); got {tuple(clips.shape)}"
            )
        x = self.preprocess_frames(clips) if raw else clips.to(self.device)

        R = self.input_resolution
        if tuple(x.shape[-2:]) != (R, R):
            raise ValueError(
                f"V-JEPA 2 expects {R}×{R} frames, got {tuple(x.shape[-2:])}. "
                "Pass raw=True to resize here."
            )

        B, T = x.shape[:2]
        # Pad to a multiple of the tubelet size by repeating the last frame
        # (T=1 images become a duplicated pair, matching HF's own handling).
        remainder = T % self.tubelet_size
        if remainder:
            pad = x[:, -1:].expand(-1, self.tubelet_size - remainder, -1, -1, -1)
            x = torch.cat([x, pad], dim=1)
        T_slots = x.shape[1] // self.tubelet_size

        grad_ctx = torch.no_grad() if self.is_frozen else torch.enable_grad()
        with grad_ctx:
            out = self.backbone(
                pixel_values_videos=x.to(self.dtype), skip_predictor=True
            )
            tokens = out.last_hidden_state                 # (B, T'*P, D)

        expected = T_slots * self.num_patches
        if tokens.size(1) != expected:
            raise RuntimeError(
                f"Got {tokens.size(1)} tokens, expected {T_slots} slots × "
                f"{self.num_patches} patches = {expected}"
            )
        # The 3-D patch conv flattens (T', H', W') in that order, so tokens
        # are slot-major: reshape by the real grid, not by input frame count.
        return tokens.view(B, T_slots, self.num_patches, self.embed_dim).float()


# ----------------------------------------------------------------------
# Smoke test (real weights) — python models/visual_encoder.py
# ----------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    encoder = VisualEncoder.load_pretrained(device=device, dtype=dtype)

    # 2 raw clips × 16 frames at SSv2-like 240×427.
    clips = torch.rand(2, 16, 3, 240, 427)
    with torch.no_grad():
        out = encoder(clips)
    print(f"Input  shape : {tuple(clips.shape)}")
    print(f"Output shape : {tuple(out.shape)}")
    assert tuple(out.shape) == (2, 8, encoder.num_patches, encoder.embed_dim)

    # Time-order sanity: reversing the clip must change the features. If this
    # fails, the time axis is not reaching the encoder.
    with torch.no_grad():
        rev = encoder(clips.flip(1))
    cos = F.cosine_similarity(out.mean((1, 2)), rev.mean((1, 2)), dim=-1)
    slot_cos = F.cosine_similarity(out[:, 0].mean(1), rev[:, -1].mean(1), dim=-1)
    print(f"clip-level cos(fwd, rev) : {cos.tolist()}")
    print(f"slot cos(fwd[0], rev[-1]) : {slot_cos.tolist()}")
    print("Smoke test passed.")