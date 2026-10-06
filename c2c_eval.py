"""
Re-score cached probe outputs with C2C's OWN evaluation procedure.

Faithful port of `Evaluator` in C2C's codes/test.py
(https://github.com/RongchangLi/ZSCAR_C2C), which itself follows the
standard CZSL evaluator (attributes-as-operators / ExplainableML czsl).
Same candidate set (train ∪ test pairs, closed world), same bias list
(built from the score gaps of correctly-classified unseen clips, ~20 bins
+ the max-bias point), same AUC (np.trapz of seen vs unseen), same HM.

Why: probe.py uses a denser bias sweep in log space. Numbers we put next to
published ones must come from the same procedure the papers used.

Usage:
    python c2c_eval.py --verb vjepa --obj clip
    python c2c_eval.py --verb vjepa --obj clip --score prob   # P(v)*P(o), C2C's CLF-model form
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from scipy.stats import hmean

from data.sth_com import SthCom
from probe import align


def c2c_metrics(scores: torch.Tensor, verb_gt, obj_gt, pair_gt, pairs_vo,
                seen_mask: torch.Tensor, closed_mask: torch.Tensor,
                train_pairs_vo: set) -> dict:
    """scores: (N, P) over ALL dataset pairs (C2C's dset.pairs order)."""
    N = scores.size(0)
    pair_list = [tuple(p) for p in pairs_vo.tolist()]
    is_seen = torch.tensor([(v, o) in train_pairs_vo
                            for v, o in zip(verb_gt.tolist(), obj_gt.tolist())])
    seen_ind, unseen_ind = is_seen.nonzero().squeeze(1), (~is_seen).nonzero().squeeze(1)

    def closed_pred(s, bias):
        s = s.clone()
        s[:, ~seen_mask] += bias
        s[:, ~closed_mask] = -1e10
        p = s.argmax(1)
        return pairs_vo[p, 0], pairs_vo[p, 1]

    def match(pred):
        return ((pred[0] == verb_gt) & (pred[1] == obj_gt)).float()

    # Reference predictions at max bias (bias=1e3), as in C2C's test().
    m_max = match(closed_pred(scores, 1e3))
    m_ub = match(closed_pred(scores, 0.0))
    seen_match_max = m_max[seen_ind].mean().item()
    unseen_match_max = m_max[unseen_ind].mean().item()

    # Bias list from correctly-classified unseen clips (at max bias).
    correct_scores = scores[torch.arange(N), pair_gt][unseen_ind]
    max_seen_scores = scores[unseen_ind][:, seen_mask].max(1)[0]
    diff = max_seen_scores - correct_scores
    correct_diff = torch.sort(diff[m_max[unseen_ind].bool()] - 1e-4)[0]
    bias_skip = max(len(correct_diff) // 20, 1)
    biaslist = correct_diff[::bias_skip]

    seen_acc, unseen_acc = [], []
    for b in biaslist.tolist():
        m = match(closed_pred(scores, b))
        seen_acc.append(m[seen_ind].mean().item())
        unseen_acc.append(m[unseen_ind].mean().item())
    seen_acc.append(seen_match_max)
    unseen_acc.append(unseen_match_max)
    seen_acc, unseen_acc = np.array(seen_acc), np.array(unseen_acc)

    hm = hmean([seen_acc, unseen_acc], axis=0)
    idx = int(np.argmax(hm))
    return {
        # np.trapz was renamed np.trapezoid in numpy 2.x; same computation.
        "AUC": float(getattr(np, "trapezoid", getattr(np, "trapz", None))(seen_acc, unseen_acc)),
        "best_hm": float(hm[idx]),
        "hm_seen": float(seen_acc[idx]), "hm_unseen": float(unseen_acc[idx]),
        "best_seen": float(seen_acc.max()), "best_unseen": float(unseen_acc.max()),
        "closed_ub_seen": m_ub[seen_ind].mean().item(),
        "closed_ub_unseen": m_ub[unseen_ind].mean().item(),
        "n_bias": len(biaslist) + 1,
    }


def run(verb_out: dict, obj_out: dict, meta: SthCom, score: str = "logprob") -> dict:
    v, o = align(verb_out["test"], obj_out["test"])
    pairs_vo = torch.tensor([[meta.verb2idx[a], meta.obj2idx[b]] for a, b in meta.pairs])
    if score == "prob":
        s = v["logp"].exp()[:, pairs_vo[:, 0]] * o["logp"].exp()[:, pairs_vo[:, 1]]
    else:
        s = v["logp"][:, pairs_vo[:, 0]] + o["logp"][:, pairs_vo[:, 1]]

    train_set = set(meta.train_pairs)
    test_set = {meta.pairs[i] for i in v["pair_idx"].tolist()}
    seen_mask = torch.tensor([p in train_set for p in meta.pairs])
    closed_mask = torch.tensor([p in (train_set | test_set) for p in meta.pairs])
    train_vo = {(meta.verb2idx[a], meta.obj2idx[b]) for a, b in meta.train_pairs}
    out = c2c_metrics(s, v["verb_idx"], v["obj_idx"], v["pair_idx"], pairs_vo,
                      seen_mask, closed_mask, train_vo)
    out["n_candidates"] = int(closed_mask.sum())
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--verb", required=True)
    p.add_argument("--obj", required=True)
    p.add_argument("--probe-root", default=os.environ.get("PROBE_ROOT", "probe_runs"))
    p.add_argument("--score", choices=["logprob", "prob"], default="logprob")
    a = p.parse_args()

    root = Path(a.probe_root)
    verb_out = torch.load(root / f"{a.verb}_verb.pt", weights_only=False)
    obj_out = torch.load(root / f"{a.obj}_obj.pt", weights_only=False)
    meta = SthCom("test", limit=1)
    r = run(verb_out, obj_out, meta, a.score)
    print(f"\nC2C evaluator | verb={a.verb} obj={a.obj} | scores={a.score} | "
          f"{r['n_candidates']} closed-world candidates, {r['n_bias']} bias points")
    print(f"  best HM {100*r['best_hm']:.2f}  AUC {100*r['AUC']:.2f}  "
          f"(S {100*r['hm_seen']:.2f} / U {100*r['hm_unseen']:.2f})")
    print(f"  best seen {100*r['best_seen']:.2f}  best unseen {100*r['best_unseen']:.2f}")


if __name__ == "__main__":
    main()