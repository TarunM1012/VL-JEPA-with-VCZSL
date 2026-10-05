"""Offline tests for the Sth-com dataset — tiny fake splits + synthetic videos."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
av = pytest.importorskip("av")

from data.sth_com import SthCom, sample_indices, verb_text  # noqa: E402


def _write_video(path, n_frames, h=48, w=80):
    with av.open(str(path), "w") as c:
        s = c.add_stream("mpeg4", rate=12)
        s.width, s.height, s.pix_fmt = w, h, "yuv420p"
        for i in range(n_frames):
            img = np.full((h, w, 3), (i * 20) % 256, dtype=np.uint8)  # brightness encodes time
            for pkt in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)


@pytest.fixture(scope="module")
def roots(tmp_path_factory):
    root = tmp_path_factory.mktemp("sthcom")
    vids, splits = root / "videos", root / "splits"
    vids.mkdir(); splits.mkdir()
    rows = {
        "train": [("1", "Opening [something]", "book"), ("2", "Closing [something]", "door")],
        "val": [("3", "Opening [something]", "door")],
        "test": [("4", "Closing [something]", "book"), ("5", "Opening [something]", "book")],
    }
    for phase, items in rows.items():
        json.dump([{"id": i, "action": "x", "verb": v, "object": o} for i, v, o in items],
                  open(splits / f"{phase}_pairs.json", "w"))
        for i, *_ in items:
            _write_video(vids / f"{i}.mp4", n_frames=10 if i != "5" else 4)
    return vids, splits


def make(roots, phase, **kw):
    vids, splits = roots
    return SthCom(phase, num_frames=8, size=32, video_root=vids,
                  split_root=splits, video_ext=".mp4", **kw)


def test_vocab_consistent_across_phases(roots):
    a, b = make(roots, "train"), make(roots, "test")
    assert a.verbs == b.verbs and a.objs == b.objs and a.pairs == b.pairs
    assert len(a.verbs) == 2 and len(a.objs) == 2


def test_seen_flags_and_candidates(roots):
    ds = make(roots, "test")
    flags = {ds[i]["video_id"]: ds[i]["is_seen"] for i in range(len(ds))}
    assert flags == {"4": False, "5": True}
    assert set(ds.closed_world_pairs) == {
        ("Opening [something]", "book"), ("Closing [something]", "door"),
        ("Closing [something]", "book")}


def test_sample_shape_and_range(roots):
    s = make(roots, "train")[0]
    assert s["frames"].shape == (8, 3, 32, 32)
    assert 0.0 <= s["frames"].min() and s["frames"].max() <= 1.0


def test_frames_in_temporal_order(roots):
    ds = make(roots, "test")
    for i in range(len(ds)):        # includes the 4-frame (short) video
        lum = ds[i]["frames"].mean(dim=(1, 2, 3))
        assert torch.all(lum[1:] >= lum[:-1] - 0.02)


def test_sample_indices():
    assert sample_indices(16, 16, jitter=False) == list(range(16))
    assert sample_indices(4, 8, jitter=False) == [0, 0, 1, 1, 2, 2, 3, 3]
    idx = sample_indices(100, 8, jitter=True)
    assert idx == sorted(idx) and all(0 <= j < 100 for j in idx)


def test_verb_text():
    assert verb_text("Putting [something] into [something]") == "putting something into something"


def test_dataloader_batches(roots):
    batch = next(iter(torch.utils.data.DataLoader(make(roots, "train"), batch_size=2)))
    assert batch["frames"].shape == (2, 8, 3, 32, 32)
    assert batch["verb_idx"].shape == (2,)