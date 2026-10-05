"""
Offline tests for CLIPEncoder — no weights needed.

Builds a tiny randomly initialised CLIP with the same architecture class as
the real one, so shape logic, chunking and encode_image parity are checked
anywhere (laptop, login node) without the 900 MB ViT-L/14 checkpoint.

    pytest tests/test_clip_encoder.py
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

clip_model_mod = pytest.importorskip("clip.model")

from models.clip_encoder import CLIPEncoder  # noqa: E402

RES, PATCH, WIDTH, EMBED = 32, 8, 64, 48      # 4×4 = 16 patches


@pytest.fixture(scope="module")
def encoder():
    torch.manual_seed(0)
    model = clip_model_mod.CLIP(
        embed_dim=EMBED, image_resolution=RES, vision_layers=2,
        vision_width=WIDTH, vision_patch_size=PATCH, context_length=77,
        vocab_size=49408, transformer_width=64, transformer_heads=2,
        transformer_layers=2,
    ).float()
    return CLIPEncoder(model)


def test_derived_dims(encoder):
    assert encoder.visual_dim == WIDTH
    assert encoder.input_resolution == RES
    assert encoder.num_patches == (RES // PATCH) ** 2
    assert encoder.text_dim == EMBED


def test_preprocess_resizes_and_keeps_leading_dims(encoder):
    raw = torch.rand(2, 5, 3, 48, 64)             # non-square, wrong size
    out = encoder.preprocess_frames(raw)
    assert out.shape == (2, 5, 3, RES, RES)


def test_preprocess_no_resize_when_already_right_size(encoder):
    raw = torch.rand(4, 3, RES, RES)
    out = encoder.preprocess_frames(raw)
    expected = (raw - encoder._mean) / encoder._std
    assert torch.allclose(out, expected, atol=1e-6)


def test_wrong_size_raises_clear_error(encoder):
    with pytest.raises(ValueError, match="preprocess_frames"):
        encoder.get_visual_features(torch.randn(2, 3, RES + 8, RES + 8))


def test_video_shapes(encoder):
    clips = torch.rand(3, 6, 3, 40, 40)
    patches, embeds = encoder.get_video_features(clips)
    assert patches.shape == (3, 6, encoder.num_patches, WIDTH)
    assert embeds.shape == (3, 6, EMBED)


def test_chunking_is_exact(encoder):
    clips = torch.rand(3, 6, 3, 40, 40)           # 18 frames
    p_big, e_big = encoder.get_video_features(clips, frame_chunk=1000)
    p_small, e_small = encoder.get_video_features(clips, frame_chunk=5)
    assert torch.allclose(p_big, p_small, atol=1e-5)
    assert torch.allclose(e_big, e_small, atol=1e-5)


def test_frame_order_preserved(encoder):
    # Frame t of clip b must come back at [b, t] — a wrong reshape would
    # silently scramble time, which is exactly what verb heads depend on.
    clips = torch.rand(2, 4, 3, RES, RES)
    _, embeds = encoder.get_video_features(clips, raw=False, frame_chunk=3)
    for b in range(2):
        for t in range(4):
            _, single = encoder.get_visual_features(clips[b, t:t + 1])
            assert torch.allclose(embeds[b, t], single[0], atol=1e-5)


def test_parity_with_encode_image(encoder):
    clips = torch.rand(2, 4, 3, RES, RES)
    _, embeds = encoder.get_video_features(clips)
    with torch.no_grad():
        ref = encoder.clip_model.encode_image(
            encoder.preprocess_frames(clips.reshape(-1, 3, RES, RES))
        ).view(2, 4, -1)
    assert torch.allclose(embeds, ref, atol=1e-5)


def test_skip_patch_tokens(encoder):
    patches, embeds = encoder.get_video_features(
        torch.rand(1, 2, 3, RES, RES), return_patch_tokens=False
    )
    assert patches is None and embeds.shape == (1, 2, EMBED)


def test_train_keeps_backbone_in_eval(encoder):
    encoder.train()
    assert not encoder.clip_model.training
    assert all(not p.requires_grad for p in encoder.clip_model.parameters())


def test_text_features(encoder):
    out = encoder.get_text_features(["opening book", "closing door"])
    assert out.shape == (2, EMBED)