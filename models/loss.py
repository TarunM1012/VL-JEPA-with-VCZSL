"""
InfoNCE loss for VL-JEPA.

Both the predictor and the Y-encoder output L2-normalised embeddings in a
shared 1536-dim space, so cosine similarity (= dot product for unit vectors)
is the natural similarity measure.  InfoNCE turns that similarity into a
contrastive objective: for each sample in a batch, the model must pick the
correct positive pair out of all negatives supplied by the other batch items.
This is identical to the CLIP loss.
"""

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Temperature is clamped to [log(1/100), log(1/0.01)] = [-log(100), log(100)]
# — the same bounds used by CLIP and adopted broadly in contrastive learning.
# The lower bound stops the distribution from collapsing to a one-hot
# (infinite sharpness); the upper bound stops it from becoming uniform
# (zero discriminability).
_TEMP_MIN = math.log(1 / 100)   # ≈ -4.605  →  temperature ≈ 100  (very soft)
_TEMP_MAX = math.log(1 / 0.01)  # ≈  4.605  →  temperature ≈ 0.01 (very sharp)


class InfoNCELoss(nn.Module):
    """
    Bidirectional InfoNCE (symmetric CLIP-style) contrastive loss.

    Given a batch of (pred, target) pairs where both tensors are L2-normalised
    and live in the same embedding space, the loss encourages:
      - pred[i] to be closer to target[i] than to any other target[j≠i]
      - target[i] to be closer to pred[i] than to any other pred[j≠i]

    The two directions are averaged into one scalar so that gradients flow
    symmetrically through both the predictor and the Y-encoder projection head.

    Temperature is a single learnable scalar stored as log(1/τ) (log-inverse
    temperature, following the CLIP convention).  Optimising in log-space keeps
    the parameter unconstrained during gradient steps while the exponentiated
    value remains strictly positive.

    Args:
        init_tau : Initial temperature τ.  Default 0.07 matches CLIP's value
                   and is a well-tested starting point for vision-language
                   contrastive training.
    """

    def __init__(self, init_tau: float = 0.07) -> None:
        super().__init__()

        # Store log(1/τ) so that τ = exp(-log_inv_tau).
        # nn.Parameter registers it for optimiser updates and state_dict saving.
        # Starting at log(1/0.07) ≈ 2.66 places τ in a range where the
        # softmax distribution is neither too peaked nor too flat at the start
        # of training.
        init_value = math.log(1.0 / init_tau)
        self.log_inv_tau = nn.Parameter(torch.tensor(init_value))

        logger.info(
            "InfoNCELoss: init τ=%.4f  (log_inv_tau=%.4f)", init_tau, init_value
        )

    @property
    def tau(self) -> torch.Tensor:
        """Current temperature τ as a scalar tensor."""
        # Clamp log_inv_tau before exponentiating so τ stays in a stable range.
        # The clamp is applied to the parameter value at read time — it does not
        # modify the stored parameter itself, preserving gradient flow through
        # values that would otherwise be out of range.
        return torch.exp(-self.log_inv_tau.clamp(_TEMP_MIN, _TEMP_MAX))

    def forward(
        self,
        pred: torch.Tensor,    # (B, 1536)  predictor output, L2-normalised
        target: torch.Tensor,  # (B, 1536)  Y-encoder output, L2-normalised
    ) -> torch.Tensor:
        """
        Args:
            pred   : L2-normalised predictor embeddings, shape (B, D).
            target : L2-normalised Y-encoder embeddings, shape (B, D).

        Returns:
            Scalar mean of the pred→target and target→pred InfoNCE losses.
        """
        B = pred.size(0)

        # ---- Similarity matrix -------------------------------------------
        # Because both tensors are L2-normalised, the dot product equals
        # cosine similarity.  The result is a (B, B) matrix where entry
        # [i, j] = cos(pred_i, target_j).
        # Diagonal entries are the positive pairs (same sample index);
        # off-diagonal entries are the negatives (different samples).
        logits = pred @ target.T   # (B, B),  values in [-1, 1]

        # ---- Temperature scaling -----------------------------------------
        # Dividing by τ sharpens (τ < 1) or softens (τ > 1) the distribution
        # before the softmax in cross-entropy.  A small τ makes the model
        # treat close-but-wrong neighbours as strongly negative; this drives
        # tighter clusters in embedding space.
        logits = logits / self.tau   # (B, B)

        # ---- Targets: the diagonal is the correct class ------------------
        # For row i of the pred→target direction, the correct target is index i.
        # Labels are therefore [0, 1, 2, ..., B-1].
        labels = torch.arange(B, device=pred.device)   # (B,)

        # ---- Bidirectional cross-entropy ---------------------------------
        # pred→target: each pred embedding must identify its paired target.
        #   rows of logits = pred queries, columns = target keys.
        loss_p2t = F.cross_entropy(logits,   labels)

        # target→pred: each target embedding must identify its paired pred.
        #   transpose flips rows↔columns, so now rows = target queries.
        loss_t2p = F.cross_entropy(logits.T, labels)

        # Average the two directions so gradients are symmetric.
        # Using the mean (rather than sum) keeps the loss magnitude
        # independent of which direction is considered "primary".
        return (loss_p2t + loss_t2p) / 2.0


# ----------------------------------------------------------------------
# Smoke test  —  python models/loss.py --smoke
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run sanity checks and exit.")
    args = parser.parse_args()

    if not args.smoke:
        parser.print_help()
        raise SystemExit(0)

    torch.manual_seed(0)
    B, D = 8, 1536

    loss_fn = InfoNCELoss()

    # ---- Check 1: perfect predictions ------------------------------------
    # When pred == target the positive logit is 1.0 (maximum cosine similarity)
    # and all negatives are random values ≤ 1.0.  Loss should be low and well
    # below log(B) (the loss of a random uniform classifier).
    embeddings = F.normalize(torch.randn(B, D), dim=-1)
    loss_perfect = loss_fn(embeddings, embeddings)
    assert loss_perfect.shape == torch.Size([]), \
        f"Expected scalar, got {loss_perfect.shape}"
    assert loss_perfect.item() < math.log(B), \
        f"Perfect-pair loss {loss_perfect.item():.4f} should be < log(B)={math.log(B):.4f}"
    print(f"[check 1] perfect pairs  → loss={loss_perfect.item():.4f}  (< log(B)={math.log(B):.2f}) ✓")

    # ---- Check 2: shuffled predictions (random negatives) ----------------
    # When pred and target are independently drawn and unrelated, the loss
    # should be close to log(B) — the model can do no better than chance.
    pred_rand   = F.normalize(torch.randn(B, D), dim=-1)
    target_rand = F.normalize(torch.randn(B, D), dim=-1)
    loss_random = loss_fn(pred_rand, target_rand)
    print(f"[check 2] random pairs   → loss={loss_random.item():.4f}  (≈ log(B)={math.log(B):.2f})")

    # ---- Check 3: loss_perfect < loss_random -----------------------------
    assert loss_perfect.item() < loss_random.item(), \
        "Perfect-pair loss should be strictly less than random-pair loss"
    print(f"[check 3] perfect < random: {loss_perfect.item():.4f} < {loss_random.item():.4f} ✓")

    # ---- Check 4: temperature is learnable -------------------------------
    assert loss_fn.log_inv_tau.requires_grad, "log_inv_tau must be a learnable parameter"
    print(f"[check 4] log_inv_tau requires_grad ✓  (τ={loss_fn.tau.item():.4f})")

    print("\nAll checks passed.")
