"""Offline tests for the VideoMAE baseline encoder — tiny random config."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
transformers = pytest.importorskip("transformers")

from extract_features import make_vjepa_fn  # noqa: E402
from models.videomae_encoder import VideoMAEEncoder  # noqa: E402


@pytest.fixture(scope="module")
def enc():
    torch.manual_seed(0)
    cfg = transformers.VideoMAEConfig(
        image_size=32, patch_size=8, num_frames=8, tubelet_size=2, hidden_size=64,
        num_hidden_layers=1, num_attention_heads=4, intermediate_size=128, use_mean_pooling=False)
    return VideoMAEEncoder.from_config(cfg)


def test_shape_from_raw_frames(enc):
    out = enc(torch.rand(2, 8, 3, 40, 70))
    assert out.shape == (2, 4, 16, 64)


def test_wrong_frame_count_raises(enc):
    with pytest.raises(ValueError, match="exactly 8 frames"):
        enc(torch.rand(1, 6, 3, 32, 32))


def test_slot_major_layout(enc):
    # Changing only frames 2–3 must change only slot 1 at the patch-embedding layer.
    x = torch.rand(1, 8, 3, 32, 32)
    x2 = x.clone(); x2[:, 2:4] = torch.rand(1, 2, 3, 32, 32)
    emb = enc.backbone.embeddings.patch_embeddings
    with torch.no_grad():
        a = emb(enc.preprocess_frames(x)).view(1, 4, 16, 64)
        b = emb(enc.preprocess_frames(x2)).view(1, 4, 16, 64)
    changed = [(a[:, s] - b[:, s]).abs().max().item() > 1e-6 for s in range(4)]
    assert changed == [False, True, False, False]


def test_pooling_reused(enc):
    out = make_vjepa_fn(enc, grid=2)(torch.rand(3, 8, 3, 32, 32))
    assert out.shape == (3, 4, 4, 64) and out.dtype == torch.float16


def test_frozen(enc):
    enc.train()
    assert not enc.backbone.training
    assert all(not p.requires_grad for p in enc.backbone.parameters())