"""
Offline tests for VisualEncoder — no weights needed.

Uses a tiny randomly initialised V-JEPA 2 (same HF class as the real model),
so the tubelet reshape, padding and preprocessing are checked anywhere.

    pytest tests/test_visual_encoder.py
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

transformers = pytest.importorskip("transformers")

from models.visual_encoder import VisualEncoder  # noqa: E402

RES, PATCH, TUBELET, DIM = 32, 8, 2, 64           # 4×4 = 16 patches per slot


@pytest.fixture(scope="module")
def encoder():
    torch.manual_seed(0)
    cfg = transformers.VJEPA2Config(
        crop_size=RES, image_size=RES, patch_size=PATCH, tubelet_size=TUBELET,
        frames_per_clip=16, hidden_size=DIM, num_hidden_layers=2,
        num_attention_heads=4, mlp_ratio=2.0,
        pred_hidden_size=32, pred_num_hidden_layers=1, pred_num_attention_heads=2,
    )
    return VisualEncoder.from_config(cfg)


def test_dims(encoder):
    assert encoder.num_patches == (RES // PATCH) ** 2
    assert encoder.tubelet_size == TUBELET
    assert encoder.embed_dim == DIM


@pytest.mark.parametrize("T, slots", [(16, 8), (8, 4), (2, 1), (1, 1), (5, 3)])
def test_output_slots(encoder, T, slots):
    out = encoder(torch.rand(2, T, 3, RES, RES))
    assert out.shape == (2, slots, encoder.num_patches, DIM)
    assert encoder.num_slots(T) == slots


def test_raw_frames_any_size(encoder):
    out = encoder(torch.rand(1, 4, 3, 30, 53))     # SSv2-like aspect ratio
    assert out.shape == (1, 2, encoder.num_patches, DIM)


def test_preprocess_shape_and_norm(encoder):
    x = torch.rand(2, 3, 3, RES, RES)
    out = encoder.preprocess_frames(x)
    assert out.shape == x.shape
    assert torch.allclose(out, (x - encoder._mean) / encoder._std, atol=1e-6)


def test_wrong_size_without_raw_raises(encoder):
    with pytest.raises(ValueError, match="raw=True"):
        encoder(torch.rand(1, 4, 3, RES + 8, RES + 8), raw=False)


def test_token_layout_is_slot_major(encoder):
    # The reshape assumes the 3-D patch conv emits tokens slot-major
    # (T', H', W'). Check it on the embedding layer, before attention mixes
    # tokens: changing only frames 2–3 must change only slot 1.
    x = torch.rand(1, 6, 3, RES, RES)
    x2 = x.clone()
    x2[:, 2:4] = torch.rand(1, 2, 3, RES, RES)
    emb_layer = encoder.backbone.encoder.embeddings
    with torch.no_grad():
        e1 = emb_layer(encoder.preprocess_frames(x)).view(1, 3, encoder.num_patches, DIM)
        e2 = emb_layer(encoder.preprocess_frames(x2)).view(1, 3, encoder.num_patches, DIM)
    changed = [(e1[:, s] - e2[:, s]).abs().max().item() > 1e-6 for s in range(3)]
    assert changed == [False, True, False]


def test_odd_frames_pad_with_last_frame(encoder):
    x = torch.rand(1, 5, 3, RES, RES)
    padded = torch.cat([x, x[:, -1:]], dim=1)
    assert torch.allclose(encoder(x), encoder(padded), atol=1e-5)


def test_reversal_changes_features(encoder):
    x = torch.rand(2, 8, 3, RES, RES)
    assert not torch.allclose(encoder(x), encoder(x.flip(1)), atol=1e-4)


def test_frozen(encoder):
    encoder.train()
    assert not encoder.backbone.training
    assert all(not p.requires_grad for p in encoder.backbone.parameters())
    out = encoder(torch.rand(1, 2, 3, RES, RES))
    assert not out.requires_grad


def test_predictor_skipped(encoder, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("predictor should not run")
    monkeypatch.setattr(encoder.backbone.predictor, "forward", boom)
    encoder(torch.rand(1, 2, 3, RES, RES))