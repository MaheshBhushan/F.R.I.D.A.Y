"""Train an openWakeWord classification head and export it to ONNX.

Architecture is copied exactly from openwakeword.train.Model's "dnn" branch --
Flatten -> Linear -> LayerNorm -> ReLU -> N x (Linear -> LayerNorm -> ReLU) ->
Linear -> Sigmoid. Not approximated: the runtime loads this ONNX with the same
[1, 16, 96] contract as the shipped alexa model, so a different graph shape or
activation order gives a model the runtime will happily run and never trust.

Negatives come from openWakeWord's published precomputed features (1000+ hours
of speech and noise, stored as a flat stream of 96-dim embeddings). Windows are
sliced from that stream rather than recomputed from audio, which is what makes
this trainable on a CPU in minutes instead of days on a GPU.

The objective is deliberately asymmetric. A wake word that misses one in twenty
tries is mildly annoying; one that fires by itself during a meeting is a thing
you switch off permanently. So negatives carry a heavier loss weight, and the
checkpoint that gets kept is chosen by false-accepts-per-hour at a fixed recall
floor, never by accuracy -- with 100x more negatives than positives, accuracy
is maximised by a model that never fires at all.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

EMBEDDINGS, EMBEDDING_DIM = 16, 96
SECONDS_PER_EMBEDDING = 0.08  # 8 mel frames at a 10ms hop


class FCNBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fcn_layer = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.relu(self.layer_norm(self.fcn_layer(x)))


class Net(nn.Module):
    def __init__(self, layer_dim: int = 128, n_blocks: int = 1) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(EMBEDDINGS * EMBEDDING_DIM, layer_dim)
        self.relu1 = nn.ReLU()
        self.layernorm1 = nn.LayerNorm(layer_dim)
        self.blocks = nn.ModuleList([FCNBlock(layer_dim) for _ in range(n_blocks)])
        self.last_layer = nn.Linear(layer_dim, 1)
        self.last_act = nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(self.layernorm1(self.layer1(self.flatten(x))))
        for block in self.blocks:
            x = block(x)
        return self.last_act(self.last_layer(x))


def sample_windows(stream: np.ndarray, count: int, rng: np.random.Generator
                   ) -> np.ndarray:
    """Slice `count` random 16-embedding windows out of a flat feature stream."""
    limit = stream.shape[0] - EMBEDDINGS
    starts = rng.integers(0, limit, size=count)
    out = np.empty((count, EMBEDDINGS, EMBEDDING_DIM), dtype=np.float32)
    for i, start in enumerate(starts):
        out[i] = stream[start:start + EMBEDDINGS]
    return out


def contiguous_windows(stream: np.ndarray, stride: int = 1) -> np.ndarray:
    """Every window in order -- the honest way to count false accepts.

    Random sampling would under-count: false accepts cluster around particular
    sounds, and skipping windows means skipping exactly the ones that fire.
    """
    n = (stream.shape[0] - EMBEDDINGS) // stride
    out = np.empty((n, EMBEDDINGS, EMBEDDING_DIM), dtype=np.float32)
    for i in range(n):
        out[i] = stream[i * stride:i * stride + EMBEDDINGS]
    return out


@torch.no_grad()
def false_accepts_per_hour(model: nn.Module, windows: np.ndarray,
                           threshold: float, batch: int = 8192) -> float:
    model.eval()
    fired = 0
    for i in range(0, len(windows), batch):
        chunk = torch.from_numpy(windows[i:i + batch])
        fired += int((model(chunk).squeeze(-1) >= threshold).sum())
    hours = len(windows) * SECONDS_PER_EMBEDDING / 3600.0
    return fired / hours if hours else float("inf")


@torch.no_grad()
def recall(model: nn.Module, positives: np.ndarray, threshold: float,
           batch: int = 8192) -> float:
    model.eval()
    hits = 0
    for i in range(0, len(positives), batch):
        chunk = torch.from_numpy(positives[i:i + batch])
        hits += int((model(chunk).squeeze(-1) >= threshold).sum())
    return hits / len(positives) if len(positives) else 0.0


def export_onnx(model: nn.Module, path: Path) -> None:
    """Export with a fixed [1, 16, 96] input, matching the shipped models.

    Batch stays 1 rather than dynamic: openWakeWord feeds exactly one window
    per frame, and a dynamic axis here has been observed to change which
    onnxruntime kernels get selected -- for a model this small that is pure
    downside.
    """
    model.eval()
    dummy = torch.zeros(1, EMBEDDINGS, EMBEDDING_DIM)
    torch.onnx.export(
        model, dummy, str(path),
        input_names=["input"], output_names=["output"],
        opset_version=13, dynamo=False,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--features", default="features")
    ap.add_argument("--negatives", default="data/negative_features_2000hrs.npy")
    ap.add_argument("--validation", default="data/validation_set_features.npy")
    ap.add_argument("--out", default="models")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--layer-dim", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--negative-weight", type=float, default=12.0,
                    help="loss weight on negatives; a false accept costs far "
                         "more than a miss")
    ap.add_argument("--negative-pool", type=int, default=400_000)
    ap.add_argument("--min-recall", type=float, default=0.50,
                    help="checkpoints below this recall are never kept, "
                         "however few false accepts they have")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    torch.set_num_threads(max(1, (torch.get_num_threads() or 8) - 2))

    feat = Path(args.features) / args.target
    pos_train = np.load(feat / "positive_train.npy")
    pos_test = np.load(feat / "positive_test.npy")
    adv_path = feat / "adversarial.npy"
    adversarial = np.load(adv_path) if adv_path.exists() else np.zeros(
        (0, EMBEDDINGS, EMBEDDING_DIM), dtype=np.float32)

    print(f"positives  train {pos_train.shape}  test {pos_test.shape}")
    print(f"adversarial      {adversarial.shape}")

    stream = np.load(args.negatives, mmap_mode="r")
    if stream.dtype != np.float32:
        print(f"negative stream dtype {stream.dtype}; casting per batch")
    print(f"negative stream  {stream.shape} "
          f"({stream.shape[0] * SECONDS_PER_EMBEDDING / 3600:.0f} hours)")
    neg_pool = sample_windows(stream, args.negative_pool, rng)

    val_stream = np.load(args.validation, mmap_mode="r")
    val_windows = contiguous_windows(val_stream)
    val_hours = len(val_windows) * SECONDS_PER_EMBEDDING / 3600
    print(f"validation  {val_windows.shape} ({val_hours:.1f} hours)")

    model = Net(args.layer_dim, args.blocks)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr * 10, total_steps=args.steps, pct_start=0.2)
    loss_fn = nn.BCELoss(reduction="none")

    # Adversarial negatives are a small pool but carry most of the signal about
    # where the decision boundary belongs, so they are oversampled relative to
    # their share of the data.
    n_pos = args.batch_size // 4
    n_adv = args.batch_size // 4 if len(adversarial) else 0
    n_neg = args.batch_size - n_pos - n_adv

    best = {"fp_per_hour": float("inf"), "recall": 0.0, "step": -1}
    best_state = None
    for step in range(1, args.steps + 1):
        model.train()
        parts = [pos_train[rng.integers(0, len(pos_train), n_pos)]]
        labels = [np.ones(n_pos, dtype=np.float32)]
        if n_adv:
            parts.append(adversarial[rng.integers(0, len(adversarial), n_adv)])
            labels.append(np.zeros(n_adv, dtype=np.float32))
        parts.append(neg_pool[rng.integers(0, len(neg_pool), n_neg)])
        labels.append(np.zeros(n_neg, dtype=np.float32))

        x = torch.from_numpy(np.concatenate(parts))
        y = torch.from_numpy(np.concatenate(labels))
        pred = model(x).squeeze(-1)
        weights = torch.where(y > 0.5, 1.0, args.negative_weight)
        loss = (loss_fn(pred, y) * weights).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

        if step % 500 == 0 or step == args.steps:
            fp = false_accepts_per_hour(model, val_windows, args.threshold)
            rec = recall(model, pos_test, args.threshold)
            print(f"step {step:>5}  loss {loss.item():.4f}  "
                  f"recall {rec:.3f}  false-accepts/hr {fp:.2f}", flush=True)
            # Recall floor first: with 100x more negatives than positives the
            # lowest-FP model is always the one that never fires.
            if rec >= args.min_recall and fp < best["fp_per_hour"]:
                best = {"fp_per_hour": fp, "recall": rec, "step": step}
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is None:
        print(f"\nno checkpoint reached the {args.min_recall:.0%} recall floor. "
              f"More positives or more training steps are needed; exporting "
              f"nothing rather than a model that cannot hear you.",
              file=sys.stderr)
        return 1

    model.load_state_dict(best_state)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    onnx_path = out / f"{args.target}.onnx"
    export_onnx(model, onnx_path)

    meta = {
        "target": args.target,
        "best_step": best["step"],
        "recall_at_threshold": round(best["recall"], 4),
        "false_accepts_per_hour": round(best["fp_per_hour"], 3),
        "threshold": args.threshold,
        "validation_hours": round(val_hours, 2),
        "positives_train": int(len(pos_train)),
        "adversarial": int(len(adversarial)),
        "negative_pool": int(len(neg_pool)),
        "layer_dim": args.layer_dim,
        "blocks": args.blocks,
        "input_shape": [1, EMBEDDINGS, EMBEDDING_DIM],
    }
    (out / f"{args.target}.json").write_text(json.dumps(meta, indent=2))
    print(f"\nexported {onnx_path}")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
