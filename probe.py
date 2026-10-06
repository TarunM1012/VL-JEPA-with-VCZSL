"""
Go/no-go probe for video CZSL on cached features.

Question: do frozen V-JEPA 2 features recognise VERBS in unseen verb–object
compositions better than frozen CLIP features?

Per backbone, two small attentive probes are trained on the train split only:
    verb probe   : tokens → 161-way logits
    object probe : tokens → 248-way logits
Composition score for a candidate pair (v, o) = log p(v) + log p(o).
Probes are cached, so a hybrid (verb from one backbone, object from another)
reuses already-trained probes.

Reported (val-selected epoch, evaluated on test):
    verb / object accuracy on seen-comp and UNSEEN-comp test clips  ← key number
    closed-world (seen ∪ test pairs, C2C protocol): best HM, AUC over a bias sweep,
        and HM at the val-chosen bias
    open-world unbiased (all 161×248 pairs, no bias, RCORE protocol): S / U / HM
    reversal: verb-prediction change rate and cosine(fwd, rev) of the verb
        embedding — a time-blind verb probe gives ~0 change and cos ≈ 1

Usage:
    python probe.py --verb vjepa --obj vjepa
    python probe.py --verb clip  --obj clip
    python probe.py --verb vjepa --obj clip      # hybrid (reuses cached probes)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.sth_com import SthCom
from extract_features import load_features

logger = logging.getLogger("probe")
_FEAT_ROOT = os.environ.get("FEATURE_ROOT", "/scratch/tarunm10/features/sth_com")


# ── features → token sequences ─────────────────────────────────────────────

def to_tokens(feats: torch.Tensor) -> torch.Tensor:
    """vjepa (N, T', G, D) → (N, T'*G, D); clip (N, T, D) unchanged."""
    return feats.flatten(1, 2) if feats.dim() == 4 else feats


def reverse_time(feats: torch.Tensor) -> torch.Tensor:
    """Exact time reversal for per-frame CLIP features (frames are independent)."""
    return feats.flip(1)


# ── probe ──────────────────────────────────────────────────────────────────

class AttentiveProbe(nn.Module):
    """Linear in → learned position embeddings → 1 transformer layer →
    learned-query cross-attention pooling → linear classifier.

    Position embeddings give the probe access to token order (time); without
    them a verb probe cannot tell forward from reversed clips.
    """

    def __init__(self, in_dim: int, num_tokens: int, num_classes: int,
                 dim: int = 512, heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, dim))
        self.pos = nn.Parameter(torch.zeros(1, num_tokens, dim))
        nn.init.normal_(self.pos, std=0.02)
        self.block = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout,
                                                activation="gelu", batch_first=True,
                                                norm_first=True)
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.query, std=0.02)
        self.pool = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(self.inp(x) + self.pos)
        q = self.query.expand(x.size(0), -1, -1)
        return self.norm(self.pool(q, x, x, need_weights=False)[0].squeeze(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x))


# ── metrics ────────────────────────────────────────────────────────────────

def pair_scores(logp_v, logp_o, pairs_vo: torch.Tensor) -> torch.Tensor:
    """(N,V),(N,O), pairs (C,2) of (v,o) idx → (N,C) log p(v)+log p(o)."""
    return logp_v[:, pairs_vo[:, 0]] + logp_o[:, pairs_vo[:, 1]]


def bias_sweep(scores, gt_col, cand_seen, gt_seen, n_bias: int = 300) -> dict:
    """Closed-world CZSL eval: add bias b to unseen candidates, sweep b.

    Uses the per-sample best seen / best unseen candidate, so each bias costs
    O(N) instead of an argmax over all candidates.
    """
    neg = torch.finfo(scores.dtype).min
    s_seen = scores.masked_fill(~cand_seen, neg)
    s_unseen = scores.masked_fill(cand_seen, neg)
    best_s, arg_s = s_seen.max(1)
    best_u, arg_u = s_unseen.max(1)
    ok_s, ok_u = arg_s == gt_col, arg_u == gt_col
    gap = best_s - best_u                        # unseen wins when b > gap
    lo, hi = gap.min().item() - 1e-3, gap.max().item() + 1e-3
    biases = torch.cat([torch.tensor([lo, 0.0, hi]), torch.linspace(lo, hi, n_bias)]).sort()[0]

    rows = []
    for b in biases.tolist():
        unseen_wins = gap < b
        correct = torch.where(unseen_wins, ok_u, ok_s)
        s = correct[gt_seen].float().mean().item()
        u = correct[~gt_seen].float().mean().item()
        rows.append((b, s, u, 2 * s * u / (s + u) if s + u > 0 else 0.0))
    best = max(rows, key=lambda r: r[3])
    pts = sorted(rows, key=lambda r: r[2])
    auc = torch.trapezoid(torch.tensor([r[1] for r in pts]),
                          torch.tensor([r[2] for r in pts])).item()
    return {"rows": rows, "best_hm": best[3], "best_bias": best[0],
            "best_seen": best[1], "best_unseen": best[2], "auc": auc}


def at_bias(sweep_rows, bias: float) -> tuple[float, float, float]:
    r = min(sweep_rows, key=lambda r: abs(r[0] - bias))
    return r[1], r[2], r[3]


# ── data plumbing ──────────────────────────────────────────────────────────

def load_split(root, backbone, split, reverse=False) -> dict:
    d = load_features(root, backbone, split, reverse=reverse)
    d["tokens"] = to_tokens(d.pop("feats"))
    return d


def align(a: dict, b: dict) -> tuple[dict, dict]:
    """Keep clips present in both caches (decode failures can differ)."""
    common = set(a["video_id"]) & set(b["video_id"])
    def sel(d):
        keep = [i for i, v in enumerate(d["video_id"]) if v in common]
        order = sorted(keep, key=lambda i: d["video_id"][i])
        idx = torch.tensor(order)
        out = {k: v[idx] for k, v in d.items() if torch.is_tensor(v)}
        out["video_id"] = [d["video_id"][i] for i in order]
        return out
    return sel(a), sel(b)


@torch.no_grad()
def predict(model, tokens, device, bs=1024, want_embed=False):
    model.eval()
    logps, embs = [], []
    for i in range(0, len(tokens), bs):
        x = tokens[i:i + bs].to(device).float()
        e = model.embed(x)
        logps.append(F.log_softmax(model.head(e), -1).cpu())
        if want_embed:
            embs.append(e.cpu())
    return torch.cat(logps), (torch.cat(embs) if want_embed else None)


def train_probe(train, val, target, num_classes, device, args) -> nn.Module:
    x, y = train["tokens"], train[f"{target}_idx"]
    model = AttentiveProbe(x.size(-1), x.size(1), num_classes, dim=args.dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = args.epochs * ((len(x) + args.batch_size - 1) // args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps, pct_start=0.1)

    best, best_state = -1.0, None
    yv, unseen_v = val[f"{target}_idx"], ~val["is_seen"].bool()
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(x))
        for i in range(0, len(x), args.batch_size):
            idx = perm[i:i + args.batch_size]
            logits = model(x[idx].to(device).float())
            loss = F.cross_entropy(logits, y[idx].to(device), label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step()
        logp, _ = predict(model, val["tokens"], device)
        correct = logp.argmax(1) == yv
        # Select on val UNSEEN-composition accuracy: that is the generalisation
        # we care about (val unseen pairs are disjoint from test unseen pairs).
        acc_u = correct[unseen_v].float().mean().item()
        logger.info("  %s ep %2d | loss %.3f | val acc all %.3f unseen-comp %.3f",
                    target, ep + 1, loss.item(), correct.float().mean().item(), acc_u)
        if acc_u > best:
            best, best_state = acc_u, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def get_probe_outputs(backbone, target, num_classes, device, args) -> dict:
    """Train (or load cached) probe → val/test log-probs (+ reversal stats for verbs)."""
    cache = Path(args.out_dir) / f"{backbone}_{target}.pt"
    if cache.exists() and not args.retrain:
        logger.info("using cached probe outputs %s", cache)
        return torch.load(cache, weights_only=False)

    logger.info("training %s probe on %s features", target, backbone)
    train = load_split(args.feat_root, backbone, "train")
    val = load_split(args.feat_root, backbone, "val")
    test = load_split(args.feat_root, backbone, "test")
    model = train_probe(train, val, target, num_classes, device, args)

    out = {}
    for name, d in (("val", val), ("test", test)):
        logp, emb = predict(model, d["tokens"], device, want_embed=(name == "test"))
        out[name] = {"logp": logp, "video_id": d["video_id"],
                     **{k: d[k] for k in ("verb_idx", "obj_idx", "pair_idx", "is_seen")}}
        if emb is not None:
            out[name]["embed"] = emb

    if target == "verb":
        rev_tokens = None
        if backbone == "clip":
            rev_tokens = reverse_time(test["tokens"])
        else:
            try:
                rev = load_split(args.feat_root, backbone, "test", reverse=True)
                fwd_ids = {v: i for i, v in enumerate(test["video_id"])}
                order = [fwd_ids.get(v) for v in rev["video_id"]]
                if None not in order and len(order) == len(test["video_id"]):
                    rev_tokens = rev["tokens"][torch.tensor(order).argsort()]
            except FileNotFoundError:
                logger.warning("no reversed features for %s; skipping reversal check", backbone)
        if rev_tokens is not None:
            logp_r, emb_r = predict(model, rev_tokens, device, want_embed=True)
            out["reversal"] = {
                "pred_change_rate": (logp_r.argmax(1) != out["test"]["logp"].argmax(1)).float().mean().item(),
                "cos_fwd_rev": F.cosine_similarity(out["test"]["embed"], emb_r).mean().item(),
            }

    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, cache)
    return out


# ── evaluation ─────────────────────────────────────────────────────────────

def evaluate(verb_out, obj_out, meta: SthCom) -> dict:
    res = {}
    if "reversal" in verb_out:
        res["reversal"] = verb_out["reversal"]

    biases = {}
    for split in ("val", "test"):
        v, o = align(verb_out[split], obj_out[split])
        logp_v, logp_o = v["logp"], o["logp"]
        gt_seen = v["is_seen"].bool()

        # Primitive accuracies split by composition seen/unseen.
        for name, logp, gt in (("verb", logp_v, v["verb_idx"]), ("obj", logp_o, v["obj_idx"])):
            c = logp.argmax(1) == gt
            res[f"{split}/{name}_acc_seen_comp"] = c[gt_seen].float().mean().item()
            res[f"{split}/{name}_acc_unseen_comp"] = c[~gt_seen].float().mean().item()

        # Closed world: train pairs ∪ this split's pairs.
        phase_pairs = sorted({meta.pairs[i] for i in v["pair_idx"].tolist()})
        cands = sorted(set(meta.train_pairs) | set(phase_pairs))
        vo = torch.tensor([[meta.verb2idx[a], meta.obj2idx[b]] for a, b in cands])
        cand_seen = torch.tensor([meta.is_seen(p) for p in cands])
        col = {p: i for i, p in enumerate(cands)}
        gt_col = torch.tensor([col[meta.pairs[i]] for i in v["pair_idx"].tolist()])
        sweep = bias_sweep(pair_scores(logp_v, logp_o, vo), gt_col, cand_seen, gt_seen)
        biases[split] = sweep
        res[f"{split}/closed_best_hm"] = sweep["best_hm"]
        res[f"{split}/closed_auc"] = sweep["auc"]
        res[f"{split}/closed_best_seen"] = sweep["best_seen"]
        res[f"{split}/closed_best_unseen"] = sweep["best_unseen"]

        # Open world, unbiased: every verb×object pair, no calibration.
        V, O = logp_v.size(1), logp_o.size(1)
        # Chunked: the full (N, V*O) matrix is ~3.6 GB for the test split.
        pred = torch.cat([
            (logp_v[i:i + 1024, :, None] + logp_o[i:i + 1024, None, :]).flatten(1).argmax(1)
            for i in range(0, len(logp_v), 1024)])
        gt_flat = v["verb_idx"] * O + v["obj_idx"]
        correct = pred == gt_flat
        s = correct[gt_seen].float().mean().item()
        u = correct[~gt_seen].float().mean().item()
        res[f"{split}/open_seen"], res[f"{split}/open_unseen"] = s, u
        res[f"{split}/open_hm"] = 2 * s * u / (s + u) if s + u else 0.0
        seen_flat = torch.zeros(V * O, dtype=torch.bool)
        for a, b in meta.train_pairs:
            seen_flat[meta.verb2idx[a] * O + meta.obj2idx[b]] = True
        res[f"{split}/open_false_seen_rate"] = seen_flat[pred[~gt_seen]].float().mean().item()

    # Bias chosen on val, applied to test (honest closed-world number).
    s, u, hm = at_bias(biases["test"]["rows"], biases["val"]["best_bias"])
    res["test/closed_hm_at_val_bias"], res["test/closed_seen_at_val_bias"], \
        res["test/closed_unseen_at_val_bias"] = hm, s, u
    return res


def print_report(name: str, r: dict) -> None:
    p = lambda k: f"{100 * r[k]:6.2f}"
    print(f"\n{'=' * 64}\n  {name}\n{'=' * 64}")
    print("  Primitive accuracy on TEST          seen-comp   UNSEEN-comp")
    print(f"    verb                               {p('test/verb_acc_seen_comp')}      {p('test/verb_acc_unseen_comp')}")
    print(f"    object                             {p('test/obj_acc_seen_comp')}      {p('test/obj_acc_unseen_comp')}")
    print("  Closed world (C2C protocol)")
    print(f"    best HM {p('test/closed_best_hm')}  AUC {p('test/closed_auc')}  "
          f"(S {p('test/closed_best_seen')} / U {p('test/closed_best_unseen')})")
    print(f"    HM at val-chosen bias {p('test/closed_hm_at_val_bias')}")
    print("  Open world, unbiased (RCORE protocol)")
    print(f"    S {p('test/open_seen')}  U {p('test/open_unseen')}  HM {p('test/open_hm')}"
          f"  | unseen clips predicted as a seen pair: {p('test/open_false_seen_rate')}")
    if "reversal" in r:
        rv = r["reversal"]
        print(f"  Reversal: verb prediction changes on {100 * rv['pred_change_rate']:.1f}% of clips,"
              f" cos(fwd, rev) = {rv['cos_fwd_rev']:.3f}")
    print("=" * 64)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--verb", choices=["vjepa", "clip", "videomae"], required=True)
    p.add_argument("--obj", choices=["vjepa", "clip", "videomae"], required=True)
    p.add_argument("--feat-root", default=_FEAT_ROOT)
    p.add_argument("--out-dir", default=os.environ.get("PROBE_ROOT", "probe_runs"))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--retrain", action="store_true")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = SthCom("test", limit=1)                     # vocab + split pairs only
    verb_out = get_probe_outputs(args.verb, "verb", len(meta.verbs), device, args)
    obj_out = get_probe_outputs(args.obj, "obj", len(meta.objs), device, args)
    res = evaluate(verb_out, obj_out, meta)
    name = f"verb={args.verb}  obj={args.obj}"
    print_report(name, res)
    out = Path(args.out_dir) / f"results_verb-{args.verb}_obj-{args.obj}.json"
    out.write_text(json.dumps(res, indent=2))
    logger.info("saved %s", out)


if __name__ == "__main__":
    main()