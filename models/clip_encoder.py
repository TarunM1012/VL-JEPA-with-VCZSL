"""
CLIP encoder: frozen visual + text backbone, usable on single images or video clips.

Design summary
--------------
Backbone   : CLIP (openai/CLIP package, NOT HuggingFace transformers).
             ViT-L/14 by default so it matches V-JEPA 2 ViT-L in size
             (~300M visual params each) for a fair backbone comparison.
Visual out : two tensors from one forward pass —
             (a) patch tokens from the final transformer block, BEFORE the CLS
                 projection layer — (B, num_patches, visual_dim);
             (b) CLIP's own image embedding (CLS → ln_post → proj),
                 (B, embed_dim), i.e. exactly what `encode_image` returns.
Video out  : CLIP is an image model, so a clip (B, T, C, H, W) is encoded frame
             by frame: frames are folded into the batch axis, encoded in
             chunks, and unfolded back to (B, T, ...). Frames never attend to
             each other, so any temporal reasoning must happen in the heads.
Text out   : CLIP's own projected text embedding — (B, embed_dim).
Both towers are frozen; forward passes run under torch.no_grad().

Input contract (the main video-related fix)
-------------------------------------------
The video dataset is shared by two backbones with different input needs:

    V-JEPA 2 : 256×256, ImageNet mean/std
    CLIP     : 224×224 (ViT-L/14), CLIP's own mean/std

So the dataset emits RAW frames (float in [0, 1], any size) and each encoder
applies its own resize + normalisation. `preprocess_frames()` does this for
CLIP. Feeding CLIP 256×256 frames directly would not just be a distribution
shift — it changes the patch grid (18×18 instead of 16×16), and the
positional-embedding add fails with a shape error. `get_visual_features()`
now checks the size up front and raises a clear error instead.
"""

from __future__ import annotations

import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_NAME = "ViT-L/14"
# Override per cluster with CLIP_DOWNLOAD_ROOT (Fir/scratch, local machine, ...).
_DEFAULT_DOWNLOAD_ROOT = os.environ.get(
    "CLIP_DOWNLOAD_ROOT", "/lustre06/project/6001346/tarunm10/.cache/clip"
)

# CLIP's normalisation statistics (from openai/CLIP `_transform`). These differ
# from the ImageNet statistics V-JEPA 2 uses.
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class CLIPEncoder(nn.Module):
    """
    Wraps a frozen CLIP model to expose:

      preprocess_frames(frames)   -> frames resized/cropped/normalised for CLIP.
      get_visual_features(images) -> (patch_tokens, image_embed) for (B, C, H, W)
          already-preprocessed images.
      get_video_features(clips)   -> (patch_tokens, image_embed) for
          (B, T, C, H, W) clips:
              patch_tokens: (B, T, num_patches, visual_dim)
              image_embed : (B, T, embed_dim)
      get_text_features(texts)    -> (B, embed_dim).

    Typical usage (video)
    ---------------------
    encoder = CLIPEncoder.load_pretrained(device=device)
    patch_tokens, frame_embeds = encoder.get_video_features(raw_clips)
    text_embeds = encoder.get_text_features(["opening book", "closing door"])
    """

    def __init__(self, clip_model: nn.Module, preprocess=None) -> None:
        super().__init__()
        self.clip_model = clip_model
        # PIL-based transform returned by clip.load(); kept for image-only
        # callers. The video path uses the tensor-based preprocess_frames().
        self.preprocess = preprocess

        visual = clip_model.visual
        # Derived from the checkpoint rather than hardcoded, so swapping to
        # ViT-B/16 (width 768) or ViT-L/14@336px needs no code change.
        self.visual_dim = visual.conv1.out_channels
        self.input_resolution = visual.input_resolution
        self.patch_size = visual.conv1.kernel_size[0]
        self.num_patches = (self.input_resolution // self.patch_size) ** 2
        self.text_dim = clip_model.text_projection.shape[1]

        self.register_buffer(
            "_mean", torch.tensor(_CLIP_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "_std", torch.tensor(_CLIP_STD).view(1, 3, 1, 1), persistent=False
        )

        for p in self.clip_model.parameters():
            p.requires_grad = False
        self.clip_model.eval()

        logger.info(
            "CLIPEncoder: frozen | input=%dpx patch=%d (%d patches) | "
            "visual_dim=%d (pre-projection) | text_dim=%d",
            self.input_resolution, self.patch_size, self.num_patches,
            self.visual_dim, self.text_dim,
        )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def load_pretrained(
        cls,
        model_name: str = _DEFAULT_MODEL_NAME,
        device: torch.device | str | None = None,
        download_root: str | None = None,
    ) -> "CLIPEncoder":
        """
        Load CLIP from a local cache via the openai/CLIP package.

        Compute nodes are offline: clip.load() only downloads when the weight
        file is missing from download_root, so pre-download it on a login node.
        """
        import clip

        root = download_root or _DEFAULT_DOWNLOAD_ROOT
        logger.info("CLIPEncoder: loading %s from %s", model_name, root)
        clip_model, preprocess = clip.load(
            model_name,
            device=device if device is not None else "cpu",
            download_root=root,
        )
        encoder = cls(clip_model, preprocess)
        # Move the wrapper too, so the mean/std buffers live on the model's
        # device (clip.load only placed the CLIP weights there).
        return encoder.to(device) if device is not None else encoder

    def train(self, mode: bool = True) -> "CLIPEncoder":
        # The backbone is frozen: keep it in eval mode even when a parent
        # module calls .train(), so no train-mode behaviour ever switches on.
        super().train(mode)
        self.clip_model.eval()
        return self

    @property
    def device(self) -> torch.device:
        return self.clip_model.visual.conv1.weight.device

    @property
    def dtype(self) -> torch.dtype:
        # fp16 on GPU (clip.load default), fp32 on CPU.
        return self.clip_model.visual.conv1.weight.dtype

    # ------------------------------------------------------------------
    # Preprocessing (tensor-based, works on batched frames)
    # ------------------------------------------------------------------

    def preprocess_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Resize (shorter side) → centre-crop → CLIP-normalise.

        Mirrors openai/CLIP's PIL `_transform` (bicubic resize of the shorter
        side to input_resolution, centre crop, normalise), but on tensors so
        it runs batched on the GPU.

        Args:
            frames: (..., 3, H, W) float in [0, 1]. Any leading dims
                (e.g. (B, T)) are preserved.
        Returns:
            (..., 3, R, R) normalised, float32, on the encoder's device.
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
                x, size=(new_h, new_w), mode="bicubic", align_corners=False,
                antialias=True,
            )
            top, left = (new_h - R) // 2, (new_w - R) // 2
            x = x[:, :, top:top + R, left:left + R]
            # Bicubic can overshoot slightly outside [0, 1].
            x = x.clamp(0.0, 1.0)

        # .to(x.device) guards against the buffers and weights ever ending up
        # on different devices (the bug that crashed the first extraction job).
        x = (x - self._mean.to(x.device)) / self._std.to(x.device)
        return x.reshape(*lead, 3, R, R)

    # ------------------------------------------------------------------
    # Visual features — images
    # ------------------------------------------------------------------

    def get_visual_features(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            images: (B, C, H, W), already CLIP-preprocessed
                (H = W = input_resolution).

        Returns:
            patch_tokens: (B, num_patches, visual_dim) — final transformer
                block, CLS dropped, before ln_post + proj.
            image_embed: (B, text_dim) — CLS through ln_post + proj, identical
                to `clip_model.encode_image(images)`. Not L2-normalised.
            Both come from one transformer pass, and both are float32.
        """
        if images.dim() != 4:
            raise ValueError(
                f"get_visual_features expects (B, C, H, W); got {tuple(images.shape)}. "
                "For clips use get_video_features()."
            )
        R = self.input_resolution
        if tuple(images.shape[-2:]) != (R, R):
            raise ValueError(
                f"CLIP expects {R}×{R} inputs, got {tuple(images.shape[-2:])}. "
                "Run preprocess_frames() first (V-JEPA 2 frames are 256×256)."
            )

        visual = self.clip_model.visual
        dtype = self.dtype

        with torch.no_grad():
            x = images.to(self.device, dtype)
            x = visual.conv1(x)                                   # (B, width, grid, grid)
            x = x.reshape(x.shape[0], x.shape[1], -1)              # (B, width, grid**2)
            x = x.permute(0, 2, 1)                                 # (B, grid**2, width)
            cls_token = visual.class_embedding.to(dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=dtype, device=x.device
            )
            x = torch.cat([cls_token, x], dim=1)                   # (B, 1+grid**2, width)
            x = x + visual.positional_embedding.to(dtype)
            x = visual.ln_pre(x)

            x = x.permute(1, 0, 2)                                 # NLD -> LND
            x = visual.transformer(x)
            x = x.permute(1, 0, 2)                                 # LND -> NLD

            patch_tokens = x[:, 1:, :]                             # drop CLS token

            # Same two lines as the tail of CLIP's VisionTransformer.forward,
            # so this matches encode_image().
            image_embed = visual.ln_post(x[:, 0, :])
            if visual.proj is not None:
                image_embed = image_embed @ visual.proj

            return patch_tokens.float(), image_embed.float()

    # ------------------------------------------------------------------
    # Visual features — video clips (frame by frame)
    # ------------------------------------------------------------------

    def get_video_features(
        self,
        clips: torch.Tensor,
        raw: bool = True,
        frame_chunk: int = 256,
        return_patch_tokens: bool = True,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """
        Encode a batch of clips frame by frame.

        Args:
            clips: (B, T, 3, H, W).
            raw: True if frames are raw [0, 1] at any size (the video dataset's
                output); they are resized/cropped/normalised here. False if
                already CLIP-preprocessed.
            frame_chunk: max frames per forward pass. B*T frames are encoded
                in chunks so a large batch of long clips cannot OOM; the
                result is identical to one big pass.
            return_patch_tokens: set False when only per-frame embeddings are
                needed (e.g. feature caching for probes) — patch tokens are
                (B, T, 256, 1024) floats and dominate memory.

        Returns:
            patch_tokens: (B, T, num_patches, visual_dim) or None.
            frame_embeds: (B, T, text_dim) — CLIP image embedding per frame.
        """
        if clips.dim() != 5:
            raise ValueError(
                f"get_video_features expects (B, T, C, H, W); got {tuple(clips.shape)}"
            )
        B, T = clips.shape[:2]
        frames = clips.reshape(B * T, *clips.shape[2:])            # (B*T, C, H, W)

        patch_chunks, embed_chunks = [], []
        for start in range(0, B * T, frame_chunk):
            chunk = frames[start:start + frame_chunk]
            if raw:
                chunk = self.preprocess_frames(chunk)
            patches, embeds = self.get_visual_features(chunk)
            if return_patch_tokens:
                patch_chunks.append(patches)
            embed_chunks.append(embeds)

        frame_embeds = torch.cat(embed_chunks, dim=0).view(B, T, -1)
        patch_tokens = None
        if return_patch_tokens:
            patch_tokens = torch.cat(patch_chunks, dim=0).view(
                B, T, self.num_patches, self.visual_dim
            )
        return patch_tokens, frame_embeds

    # ------------------------------------------------------------------
    # Text features
    # ------------------------------------------------------------------

    def get_text_features(self, texts: list[str]) -> torch.Tensor:
        """
        Args:
            texts: list of B strings.

        Returns:
            (B, text_dim) — CLIP's own projected text embedding, float32.
            Not L2-normalised (matches CLIP's native output; callers that
            need cosine similarity should normalise explicitly).
        """
        import clip

        tokens = clip.tokenize(texts, truncate=True).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(tokens)
        return text_features.float()


# ----------------------------------------------------------------------
# Smoke test (real weights) — python models/clip_encoder.py
# ----------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = CLIPEncoder.load_pretrained(device=device)

    # Raw video-style input: 2 clips × 16 frames at V-JEPA's 256×256.
    clips = torch.rand(2, 16, 3, 256, 256)
    texts = ["opening book", "closing door"]

    patch_tokens, frame_embeds = encoder.get_video_features(clips, frame_chunk=12)
    text_embeds = encoder.get_text_features(texts)

    print(f"Patch tokens shape : {tuple(patch_tokens.shape)}")
    print(f"Frame embeds shape : {tuple(frame_embeds.shape)}")
    print(f"Text embeds shape  : {tuple(text_embeds.shape)}")

    assert tuple(patch_tokens.shape) == (2, 16, encoder.num_patches, encoder.visual_dim)
    assert tuple(frame_embeds.shape) == (2, 16, encoder.text_dim)
    assert tuple(text_embeds.shape) == (2, encoder.text_dim)

    # Per-frame embeddings must equal CLIP's own encode_image on the same
    # preprocessed frames — otherwise the video path silently diverges from
    # the zero-shot CLIP the heads are compared against.
    with torch.no_grad():
        ref = encoder.clip_model.encode_image(
            encoder.preprocess_frames(clips[0]).to(encoder.dtype)
        ).float()
    max_diff = (frame_embeds[0] - ref).abs().max().item()
    print(f"max |ours - encode_image| : {max_diff:.2e}")
    # fp16 on GPU: allow a small tolerance.
    assert max_diff < (5e-3 if encoder.dtype == torch.float16 else 1e-4), max_diff
    print("Shape + parity check passed.")