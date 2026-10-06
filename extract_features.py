"""
Extract frozen backbone features for Sth-com once, cache them to disk, and
train every probe/head on the cache afterwards.

What gets saved per clip (fp16):
    videomae : same layout as vjepa, from VideoMAE ViT-L (14×14 grid → G×G)
    vjepa : (8, G*G, 1024)  V-JEPA 2 last-layer tokens, one entry per temporal
            slot (2 frames), spatially average-pooled from the 16×16 grid to
            G×G (default G=4 → 16 cells). Keeping a small grid instead of one
            mean vector matters: V-JEPA features are normally read with an
            attentive probe, and a global mean throws away "where" information
            that verbs like "pushing left to right" need.
    clip  : (16, 768)       CLIP image embedding (CLS → ln_post → proj) per
            frame. Same space as CLIP text embeddings, so zero-shot CLIP
            text matching works directly on the cache.

Output layout (resumable — finished shards are skipped on restart):
    <out_dir>/<backbone>/<split>/shard_00000.pt  {"feats", "video_id",
                                                  "verb_idx", "obj_idx",
                                                  "pair_idx", "is_seen"}
    <out_dir>/<backbone>/<split>/meta.json       config + failed video ids

Usage:
    # timing run first (1,000 clips), then the full job
    python extract_features.py --backbone vjepa --split test --limit 1000
    python extract_features.py --backbone vjepa --split train val test
    python extract_features.py --backbone clip  --split train val test

    # reversed clips for the time-order diagnostic (test set only is enough)
    python extract_features.py --backbone vjepa --split test --reverse

Load the cache with `load_features(out_dir, backbone, split)`.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from data.sth_com import SthCom

logger = logging.getLogger("extract")

_DEFAULT_OUT = os.environ.get("FEATURE_ROOT", "/scratch/tarunm10/features/sth_com")


# ── robust dataset wrapper ─────────────────────────────────────────────────

class SafeDataset(Dataset):
    """Returns None for clips that fail to decode instead of killing the job."""

    def __init__(self, ds: SthCom, reverse: bool = False) -> None:
        self.ds, self.reverse = ds, reverse

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i: int):
        try:
            s = self.ds[i]
        except Exception as exc:  # corrupt/missing video
            return {"failed": self.ds.samples[i][0], "error": repr(exc)}
        if self.reverse:
            s["frames"] = s["frames"].flip(0)
        return s


def collate(batch: list[dict]) -> dict:
    ok = [b for b in batch if "failed" not in b]
    out = {"failed": [b["failed"] for b in batch if "failed" in b]}
    if ok:
        out["frames"] = torch.stack([b["frames"] for b in ok])
        for k in ("verb_idx", "obj_idx", "pair_idx", "is_seen"):
            out[k] = torch.tensor([b[k] for b in ok])
        out["video_id"] = [b["video_id"] for b in ok]
    return out


# ── backbones → pooled features ────────────────────────────────────────────

def build_extractor(backbone: str, device: torch.device, grid: int = 4):
    """Returns fn(frames (B,T,3,S,S) in [0,1]) -> features (B, ...) on CPU fp16."""
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    if backbone == "vjepa":
        from models.visual_encoder import VisualEncoder
        enc = VisualEncoder.load_pretrained(device=device, dtype=dtype)
        return make_vjepa_fn(enc, grid)

    if backbone == "videomae":
        from models.videomae_encoder import VideoMAEEncoder
        enc = VideoMAEEncoder.load_pretrained(device=device, dtype=dtype)
        return make_vjepa_fn(enc, grid)          # same token layout → same pooling

    if backbone == "clip":
        from models.clip_encoder import CLIPEncoder
        enc = CLIPEncoder.load_pretrained(device=device)
        return make_clip_fn(enc)

    raise ValueError(f"unknown backbone {backbone!r}")


def make_vjepa_fn(enc, grid: int):
    side = enc.grid_size

    @torch.no_grad()
    def fn(frames: torch.Tensor) -> torch.Tensor:
        tok = enc(frames)                                   # (B, T', P, D)
        B, T, P, D = tok.shape
        x = tok.view(B * T, side, side, D).permute(0, 3, 1, 2)
        x = F.adaptive_avg_pool2d(x, grid)                  # (B*T', D, g, g)
        x = x.flatten(2).transpose(1, 2)                    # (B*T', g*g, D)
        return x.reshape(B, T, grid * grid, D).half().cpu()

    return fn


def make_clip_fn(enc):
    @torch.no_grad()
    def fn(frames: torch.Tensor) -> torch.Tensor:
        _, emb = enc.get_video_features(frames, return_patch_tokens=False)
        return emb.half().cpu()                             # (B, T, 768)

    return fn


# ── main loop ──────────────────────────────────────────────────────────────

def extract_split(
    dataset: SthCom,
    feature_fn,
    out_dir: Path,
    batch_size: int = 16,
    shard_size: int = 2048,
    num_workers: int = 8,
    reverse: bool = False,
    config: dict | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(dataset)
    n_shards = (n + shard_size - 1) // shard_size
    failed: list[str] = []
    done_clips, t0 = 0, time.time()

    for s in range(n_shards):
        path = out_dir / f"shard_{s:05d}.pt"
        if path.exists():
            logger.info("skip %s (exists)", path.name)
            continue
        idx = list(range(s * shard_size, min((s + 1) * shard_size, n)))
        loader = DataLoader(
            torch.utils.data.Subset(SafeDataset(dataset, reverse), idx),
            batch_size=batch_size, num_workers=num_workers, collate_fn=collate,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=False,
        )
        parts = {k: [] for k in ("feats", "verb_idx", "obj_idx", "pair_idx", "is_seen")}
        ids: list[str] = []
        for batch in loader:
            failed += batch["failed"]
            if "frames" not in batch:
                continue
            parts["feats"].append(feature_fn(batch["frames"]))
            for k in ("verb_idx", "obj_idx", "pair_idx", "is_seen"):
                parts[k].append(batch[k])
            ids += batch["video_id"]
            done_clips += len(batch["video_id"])

        shard = {k: torch.cat(v) for k, v in parts.items() if v}
        shard["video_id"] = ids
        tmp = path.with_suffix(".tmp")
        torch.save(shard, tmp)
        tmp.rename(path)                     # atomic: no half-written shards
        rate = done_clips / max(time.time() - t0, 1e-6)
        logger.info("wrote %s (%d clips) | %.1f clips/s | ETA %.1f min",
                    path.name, len(ids), rate, (n - (s + 1) * shard_size) / max(rate, 1e-6) / 60)

    meta_path = out_dir / "meta.json"
    old_failed = json.loads(meta_path.read_text()).get("failed", []) if meta_path.exists() else []
    meta = {**(config or {}), "num_clips": n, "reverse": reverse,
            "failed": sorted(set(old_failed) | set(failed))}
    meta_path.write_text(json.dumps(meta, indent=2))
    if failed:
        logger.warning("%d clips failed to decode (listed in meta.json)", len(failed))
    return meta


def load_features(out_dir: Path | str, backbone: str, split: str,
                  reverse: bool = False) -> dict:
    """Concatenate all shards → {"feats", "verb_idx", ..., "video_id"}."""
    d = Path(out_dir) / backbone / (f"{split}_reversed" if reverse else split)
    shards = sorted(d.glob("shard_*.pt"))
    if not shards:
        raise FileNotFoundError(f"no shards in {d}")
    parts = [torch.load(p, weights_only=False) for p in shards]
    parts = [p for p in parts if "feats" in p]      # shards where every clip failed
    out = {k: torch.cat([p[k] for p in parts])
           for k in ("feats", "verb_idx", "obj_idx", "pair_idx", "is_seen")}
    out["video_id"] = [v for p in parts for v in p["video_id"]]
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", choices=["vjepa", "clip", "videomae"], required=True)
    p.add_argument("--split", nargs="+", default=["train", "val", "test"])
    p.add_argument("--out-dir", default=_DEFAULT_OUT)
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--grid", type=int, default=4, help="V-JEPA spatial pool grid (G×G)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--shard-size", type=int, default=2048)
    p.add_argument("--num-workers", type=int,
                   default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    p.add_argument("--limit", type=int, default=None, help="first N clips (timing runs)")
    p.add_argument("--reverse", action="store_true", help="extract time-reversed clips")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device=%s backbone=%s splits=%s", device, args.backbone, args.split)

    feature_fn = build_extractor(args.backbone, device, grid=args.grid)
    for split in args.split:
        # No frame jitter: cached features must be deterministic.
        ds = SthCom(split, num_frames=args.num_frames, jitter=False, limit=args.limit)
        name = f"{split}_reversed" if args.reverse else split
        if args.limit:
            name += f"_limit{args.limit}"
        out = Path(args.out_dir) / args.backbone / name
        logger.info("%s: %d clips → %s", split, len(ds), out)
        extract_split(ds, feature_fn, out, batch_size=args.batch_size,
                      shard_size=args.shard_size, num_workers=args.num_workers,
                      reverse=args.reverse,
                      config={"backbone": args.backbone, "split": split,
                              "num_frames": args.num_frames, "grid": args.grid})


if __name__ == "__main__":
    main()