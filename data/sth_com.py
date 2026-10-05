"""
Sth-com (Something-composition) dataset for zero-shot compositional action
recognition (ZS-CAR). Splits from C2C (ECCV 2024):
https://github.com/RongchangLi/ZSCAR_C2C/tree/main/data_split/generalized

Each split JSON is a list of {"id", "action", "verb", "object"}:
    id     : SSv2 video id → <video_root>/<id>.webm
    verb   : SSv2 template, e.g. "Opening [something]"   (161 verbs)
    object : noun, e.g. "book"                            (248 objects)
    action : the original free-text SSv2 caption (NOT canonical — the same
             verb–object pair has many captions, so it is not used as a label)

Sizes: train 38,034 / val 18,774 / test 22,657 videos.
Test = 976 seen + 956 unseen pairs; every test verb and object is in train.

Each sample is a dict:
    frames   : (T, 3, S, S) float in [0, 1] — RAW frames. Each encoder
               (VisualEncoder / CLIPEncoder) applies its own resize + norm.
    verb_idx, obj_idx, pair_idx : int indices into the global vocab
    is_seen  : bool, pair appears in the train split
    video_id : str, for aligning cached features with labels

Frames are resized (shorter side → S) and centre-cropped to S×S here only so
clips can be batched; S defaults to 256 (V-JEPA 2's input size). CLIP then
downsizes 256 → 224 itself.

Frame sampling (TSN-style, as in C2C): split the video into T equal segments
and take one frame per segment — a random one when train-time jitter is on,
the middle one otherwise. Videos shorter than T frames repeat frames in order.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

Phase = Literal["train", "val", "test"]

_DEFAULT_VIDEO_ROOT = os.environ.get(
    "SSV2_VIDEO_ROOT", "/scratch/tarunm10/datasets/ssv2/videos"
)
_DEFAULT_SPLIT_ROOT = os.environ.get(
    "STHCOM_SPLIT_ROOT", "/scratch/tarunm10/datasets/sth_com/data_split/generalized"
)


def verb_text(template: str) -> str:
    """'Opening [something]' → 'opening something' (C2C's convention)."""
    return template.replace("[", "").replace("]", "").replace(",", "").lower()


def load_split(split_root: Path | str, phase: Phase) -> list[dict]:
    with open(Path(split_root) / f"{phase}_pairs.json") as f:
        return json.load(f)


def sample_indices(n_frames: int, num_frames: int, jitter: bool, rng=random) -> list[int]:
    """One index per equal segment; middle of segment unless jitter."""
    if n_frames <= 0:
        raise ValueError("video has no frames")
    if n_frames < num_frames:
        # Spread available frames over T slots, keeping temporal order.
        return [int(i * n_frames / num_frames) for i in range(num_frames)]
    ticks = np.linspace(0, n_frames, num_frames + 1)
    idx = []
    for i in range(num_frames):
        lo, hi = int(ticks[i]), max(int(ticks[i]) + 1, int(ticks[i + 1]))
        idx.append(rng.randrange(lo, hi) if jitter else (lo + hi - 1) // 2)
    return idx


def decode_video(path: Path | str) -> np.ndarray:
    """Decode every frame of a (short) video → (N, H, W, 3) uint8 RGB.

    SSv2 clips are 2–6 s at 12 fps and 240p, so decoding all ~40 frames and
    indexing is simpler and barely slower than seeking.
    """
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(stream)]
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    return np.stack(frames)


def resize_crop(frames: torch.Tensor, size: int) -> torch.Tensor:
    """(T, 3, H, W) float → shorter side to `size`, centre crop size×size."""
    H, W = frames.shape[-2:]
    scale = size / min(H, W)
    nh, nw = max(size, round(H * scale)), max(size, round(W * scale))
    if (nh, nw) != (H, W):
        frames = F.interpolate(frames, size=(nh, nw), mode="bilinear",
                               align_corners=False, antialias=True)
    top, left = (nh - size) // 2, (nw - size) // 2
    return frames[:, :, top:top + size, left:left + size].clamp(0, 1)


class SthCom(Dataset):
    """
    Args:
        phase       : "train" | "val" | "test"
        num_frames  : frames per clip (16 → 8 V-JEPA 2 temporal slots)
        size        : square side of returned frames
        jitter      : random frame within each segment (default: train only)
        video_root  : folder of <id>.webm (env SSV2_VIDEO_ROOT)
        split_root  : folder with {train,val,test}_pairs.json (env STHCOM_SPLIT_ROOT)
        video_ext   : file extension of the videos
        limit       : keep only the first N samples (smoke tests / timing runs)
    """

    def __init__(
        self,
        phase: Phase = "train",
        num_frames: int = 16,
        size: int = 256,
        jitter: bool | None = None,
        video_root: Path | str = _DEFAULT_VIDEO_ROOT,
        split_root: Path | str = _DEFAULT_SPLIT_ROOT,
        video_ext: str = ".webm",
        limit: int | None = None,
    ) -> None:
        self.phase = phase
        self.num_frames = num_frames
        self.size = size
        self.jitter = (phase == "train") if jitter is None else jitter
        self.video_root = Path(video_root)
        self.video_ext = video_ext

        splits = {p: load_split(split_root, p) for p in ("train", "val", "test")}

        # Global vocab over all three splits, sorted → identical indices no
        # matter which phase is loaded (same rule as the MIT-States loader).
        every = [x for items in splits.values() for x in items]
        self.verbs: list[str] = sorted({x["verb"] for x in every})
        self.objs: list[str] = sorted({x["object"] for x in every})
        self.pairs: list[tuple[str, str]] = sorted({(x["verb"], x["object"]) for x in every})
        self.verb2idx = {v: i for i, v in enumerate(self.verbs)}
        self.obj2idx = {o: i for i, o in enumerate(self.objs)}
        self.pair2idx = {p: i for i, p in enumerate(self.pairs)}

        self.train_pairs = sorted({(x["verb"], x["object"]) for x in splits["train"]})
        self.phase_pairs = sorted({(x["verb"], x["object"]) for x in splits[phase]})
        self._seen = set(self.train_pairs)

        items = splits[phase][:limit] if limit else splits[phase]
        self.samples = [(str(x["id"]), x["verb"], x["object"]) for x in items]

    # ── candidate sets for evaluation ────────────────────────────────────
    @property
    def closed_world_pairs(self) -> list[tuple[str, str]]:
        """Seen (train) pairs ∪ this phase's pairs."""
        return sorted(set(self.train_pairs) | set(self.phase_pairs))

    def is_seen(self, pair: tuple[str, str]) -> bool:
        return pair in self._seen

    @property
    def verb_texts(self) -> list[str]:
        return [verb_text(v) for v in self.verbs]

    # ── Dataset protocol ─────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.samples)

    def load_frames(self, video_id: str) -> torch.Tensor:
        video = decode_video(self.video_root / f"{video_id}{self.video_ext}")
        idx = sample_indices(len(video), self.num_frames, self.jitter)
        clip = torch.from_numpy(video[idx]).permute(0, 3, 1, 2).float().div_(255)
        return resize_crop(clip, self.size)              # (T, 3, S, S)

    def __getitem__(self, i: int) -> dict:
        video_id, verb, obj = self.samples[i]
        return {
            "frames": self.load_frames(video_id),
            "verb_idx": self.verb2idx[verb],
            "obj_idx": self.obj2idx[obj],
            "pair_idx": self.pair2idx[(verb, obj)],
            "is_seen": (verb, obj) in self._seen,
            "video_id": video_id,
        }


# ── Smoke test: python data/sth_com.py ──────────────────────────────────────
if __name__ == "__main__":
    import time

    ds = SthCom("test", limit=8)
    print(f"verbs={len(ds.verbs)} objs={len(ds.objs)} pairs={len(ds.pairs)}")
    print(f"closed-world candidates (test) = {len(ds.closed_world_pairs)}")
    t0 = time.time()
    s = ds[0]
    print(f"frames {tuple(s['frames'].shape)} in {time.time() - t0:.2f}s | "
          f"{ds.verbs[s['verb_idx']]!r} + {ds.objs[s['obj_idx']]!r} seen={s['is_seen']}")
    assert s["frames"].shape == (16, 3, 256, 256)
    assert (len(ds.verbs), len(ds.objs)) == (161, 248), "unexpected vocab size"