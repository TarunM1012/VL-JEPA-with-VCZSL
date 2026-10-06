"""Offline tests for extract_features.py — tiny fake data, tiny random encoders."""

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
av = pytest.importorskip("av")
transformers = pytest.importorskip("transformers")

from data.sth_com import SthCom  # noqa: E402
from extract_features import extract_split, load_features, make_vjepa_fn  # noqa: E402
from models.visual_encoder import VisualEncoder  # noqa: E402
from tests.test_sth_com import _write_video  # noqa: E402


@pytest.fixture()
def data(tmp_path):
    vids, splits = tmp_path / "videos", tmp_path / "splits"
    vids.mkdir(); splits.mkdir()
    rows = [(str(i), "Opening [something]" if i % 2 else "Closing [something]",
             "book" if i < 3 else "door") for i in range(7)]
    for phase in ("train", "val", "test"):
        json.dump([{"id": i, "action": "x", "verb": v, "object": o} for i, v, o in rows],
                  open(splits / f"{phase}_pairs.json", "w"))
    for i, *_ in rows:
        _write_video(vids / f"{i}.mp4", n_frames=8)
    (vids / "6.mp4").write_bytes(b"not a video")          # one corrupt clip
    ds = SthCom("test", num_frames=4, size=32, jitter=False,
                video_root=vids, split_root=splits, video_ext=".mp4")
    return ds, tmp_path / "feats"


def dummy_fn(frames):
    return frames.mean(dim=(2, 3, 4)).half()              # (B, T)


def test_shards_labels_and_failures(data):
    ds, out = data
    meta = extract_split(ds, dummy_fn, out / "dummy" / "test",
                         batch_size=2, shard_size=3, num_workers=0)
    assert meta["failed"] == ["6"]
    assert len(list((out / "dummy" / "test").glob("shard_*.pt"))) == 3
    f = load_features(out, "dummy", "test")
    assert f["feats"].shape == (6, 4)
    assert f["video_id"] == [str(i) for i in range(6)]
    for j, vid in enumerate(f["video_id"]):
        _, verb, obj = ds.samples[int(vid)]
        assert f["verb_idx"][j] == ds.verb2idx[verb]
        assert f["obj_idx"][j] == ds.obj2idx[obj]


def test_resume_skips_existing(data):
    ds, out = data
    d = out / "dummy" / "test"
    extract_split(ds, dummy_fn, d, batch_size=2, shard_size=3, num_workers=0)
    calls = []
    extract_split(ds, lambda x: calls.append(1) or dummy_fn(x), d,
                  batch_size=2, shard_size=3, num_workers=0)
    assert calls == []


def test_reverse_flips_time(data):
    ds, out = data
    extract_split(ds, dummy_fn, out / "d" / "test", batch_size=2, shard_size=10, num_workers=0)
    extract_split(ds, dummy_fn, out / "d" / "test_reversed", batch_size=2,
                  shard_size=10, num_workers=0, reverse=True)
    fwd, rev = load_features(out, "d", "test"), load_features(out, "d", "test", reverse=True)
    assert torch.allclose(fwd["feats"].float(), rev["feats"].float().flip(1), atol=1e-3)


def test_vjepa_pooling_shape():
    torch.manual_seed(0)
    cfg = transformers.VJEPA2Config(
        crop_size=32, image_size=32, patch_size=8, tubelet_size=2, hidden_size=64,
        num_hidden_layers=1, num_attention_heads=4, mlp_ratio=2.0,
        pred_hidden_size=32, pred_num_hidden_layers=1, pred_num_attention_heads=2)
    enc = VisualEncoder.from_config(cfg)                  # 4×4 patch grid
    fn = make_vjepa_fn(enc, grid=2)
    out = fn(torch.rand(3, 8, 3, 32, 32))
    assert out.shape == (3, 4, 4, 64) and out.dtype == torch.float16
    # grid == patch grid → pooling is the identity on the encoder tokens
    full = make_vjepa_fn(enc, grid=4)(x := torch.rand(1, 2, 3, 32, 32))
    assert torch.allclose(full.float(), enc(x).half().float(), atol=1e-2)