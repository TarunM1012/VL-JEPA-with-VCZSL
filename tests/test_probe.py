"""Offline tests for probe.py — metrics on hand-checkable inputs, plus an
end-to-end run on synthetic cached features where the labels are learnable."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import probe  # noqa: E402
from data.sth_com import SthCom  # noqa: E402


def test_bias_sweep_extremes():
    # 2 candidates: col 0 seen, col 1 unseen. Sample A gt seen, B gt unseen.
    scores = torch.tensor([[1.0, 0.0], [1.0, 0.5]])
    out = probe.bias_sweep(scores, torch.tensor([0, 1]),
                           torch.tensor([True, False]), torch.tensor([True, False]))
    # Some bias in (0.5, 1.0) gets both right.
    assert out["best_hm"] == 1.0 and 0.5 < out["best_bias"] < 1.0
    s0, u0, _ = probe.at_bias(out["rows"], 0.0)
    assert (s0, u0) == (1.0, 0.0)


def test_probe_uses_time_order():
    m = probe.AttentiveProbe(16, 8, 3)
    x = torch.randn(4, 8, 16)
    m.eval()
    assert not torch.allclose(m(x), m(x.flip(1)))


def _write_fake_cache(root: Path, backbone: str, meta: SthCom, n_per_split: dict):
    """Features = noisy one-hots of verb (first dims) and object (later dims),
    with the verb signal only in the time ORDER for vjepa-style 4-D features."""
    V, O = len(meta.verbs), len(meta.objs)
    g = torch.Generator().manual_seed(0)
    train_pairs = meta.train_pairs
    for split, n in n_per_split.items():
        pairs = train_pairs if split == "train" else sorted(set(meta.phase_pairs) | set(train_pairs))
        if split != "train":
            pairs = [p for p in pairs]
        idx = torch.randint(len(pairs), (n,), generator=g)
        vi = torch.tensor([meta.verb2idx[pairs[i][0]] for i in idx])
        oi = torch.tensor([meta.obj2idx[pairs[i][1]] for i in idx])
        D = V + O
        base = torch.zeros(n, D)
        base[torch.arange(n), vi] = 3.0
        base[torch.arange(n), V + oi] = 3.0
        feats = base[:, None, :].repeat(1, 4, 1) + 0.3 * torch.randn(n, 4, D, generator=g)
        if backbone == "fake4d":
            feats = feats[:, :, None, :].repeat(1, 1, 2, 1)
        d = root / backbone / split
        d.mkdir(parents=True)
        pair_idx = torch.tensor([meta.pair2idx[pairs[i]] for i in idx])
        torch.save({"feats": feats.half(), "verb_idx": vi, "obj_idx": oi,
                    "pair_idx": pair_idx,
                    "is_seen": torch.tensor([meta.is_seen(pairs[i]) for i in idx]),
                    "video_id": [f"{split}{j}" for j in range(n)]}, d / "shard_00000.pt")


def test_end_to_end_learnable(tmp_path, monkeypatch):
    splits = tmp_path / "splits"; splits.mkdir()
    verbs = ["Opening [something]", "Closing [something]", "Pushing [something]"]
    objs = ["book", "door", "box"]
    allp = [(v, o) for v in verbs for o in objs]
    # Every verb and object appears in train; pairs 1, 5, 8 are held out.
    train = [p for i, p in enumerate(allp) if i not in (1, 5, 8)]
    for phase, ps in (("train", train), ("val", train + [allp[5]]),
                      ("test", train + [allp[1], allp[8]])):
        json.dump([{"id": str(i), "action": "x", "verb": v, "object": o} for i, (v, o) in enumerate(ps)],
                  open(splits / f"{phase}_pairs.json", "w"))
    meta = SthCom("test", split_root=splits, video_root=tmp_path)
    feat_root = tmp_path / "feats"
    _write_fake_cache(feat_root, "fake4d", meta, {"train": 600, "val": 200, "test": 200})

    args = SimpleNamespace(feat_root=feat_root, out_dir=tmp_path / "probes", epochs=15,
                           batch_size=64, lr=2e-3, wd=0.0, dim=32, retrain=False)
    dev = torch.device("cpu")
    v_out = probe.get_probe_outputs("fake4d", "verb", 3, dev, args)
    o_out = probe.get_probe_outputs("fake4d", "obj", 3, dev, args)
    res = probe.evaluate(v_out, o_out, meta)
    assert res["test/verb_acc_unseen_comp"] > 0.9
    assert res["test/obj_acc_unseen_comp"] > 0.9
    assert res["test/closed_best_hm"] > 0.8
    assert 0.0 <= res["test/open_hm"] <= 1.0
    # cached: second call must not retrain
    assert (tmp_path / "probes" / "fake4d_verb.pt").exists()
    probe.print_report("fake", res)