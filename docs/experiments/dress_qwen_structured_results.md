# Dress DQU-CIR + Qwen structured-text experiments

## Evaluation protocol

- Dataset: FashionIQ dress, `original-split`
- Baseline checkpoint: `dress_dqu_official_original_seed42_best.pt`
- CLIP: LAION ViT-H/14
- DQU backbone: frozen and unchanged
- Structured data: Qwen 3.7 Flash train/validation JSON
- Training autocast: BF16 on supported GPUs (FP16 fallback)
- Retrieval evaluation autocast: FP16, matching the original DQU experiments
- Selection score: R@10 + R@50

Changing evaluation from FP16 to BF16 changes some borderline gallery ranks.
All results below therefore use FP16 evaluation.

## Sequential optimization

| Version | Main change | R@1 | R@10 | R@50 | R@10+R@50 |
|---|---|---:|---:|---:|---:|
| DQU | Verified baseline | 21.6658 | 52.1567 | 74.9132 | 127.0699 |
| v4 | Bounded effective structured residual | 21.6163 | 52.5037 | 75.0124 | 127.5161 |
| v5 | Gate warmup, then adapter fitting | 21.8642 | 52.6525 | 75.0124 | 127.6648 |
| v6 | Separate JSON field encoding | 21.8146 | 52.6029 | 75.1116 | 127.7144 |
| v7 seed 42 | Calibrated confidence + stable masked field average | 21.6163 | 52.7516 | 75.1116 | **127.8632** |
| v7 seed 43 | Same configuration | 21.7650 | 52.5533 | 75.0124 | 127.5657 |
| v7 seed 44 | Same configuration | 21.7650 | 52.3054 | 75.0620 | 127.3674 |

The learned v6 field attention had entropy 1.375 versus the four-field
uniform maximum 1.386. It contributed little selectivity and produced
non-finite gradients on some batches. v7 therefore uses a deterministic
masked average over present fields. Training uses BF16 because FP16 loss
scaling also overflowed the learned field scorer.

## Three-seed v7 result

| Metric | Mean | Sample standard deviation | DQU baseline | Mean delta |
|---|---:|---:|---:|---:|
| R@1 | 21.7154 | 0.0859 | 21.6658 | +0.0496 |
| R@10 | 52.5368 | 0.2236 | 52.1567 | +0.3801 |
| R@50 | 75.0620 | 0.0496 | 74.9132 | +0.1487 |
| R@10+R@50 | 127.5987 | 0.2500 | 127.0699 | +0.5288 |
| Mean rank | 98.6882 | 0.2944 | 99.3713 | -0.6832 |

All three seeds improve R@10+R@50 over DQU. The improvement is modest but
directionally reproducible. It should not be reported using seed 42 alone.

## Seed-42 subgroup deltas versus DQU

The same v7 checkpoint was evaluated twice, with structured text enabled and
hard-disabled. This isolates the adapter effect while retaining identical
query/gallery encoding and evaluation precision.

| Query group | n | ΔR@1 | ΔR@10 | ΔR@50 | Δ mean rank |
|---|---:|---:|---:|---:|---:|
| Has relation | 2005 | -0.0499 | +0.5985 | +0.1995 | -1.0254 |
| Add and remove | 1986 | -0.1007 | +0.6042 | +0.2014 | -1.0227 |
| High confidence (>=0.9) | 1264 | +0.1582 | +0.8703 | +0.0791 | -0.7334 |
| Lower confidence (<0.9) | 753 | -0.3984 | +0.1328 | +0.3984 | -1.4993 |
| Complex | 1620 | -0.1235 | +0.4938 | +0.3086 | -1.0932 |
| Simple | 397 | +0.2519 | +1.0076 | -0.2519 | -0.7179 |

The no-relation and add-only groups contain only 12 and 19 queries,
respectively, so their recalls are too noisy for conclusions.

## v8 residual-only hard-negative margin loss

The v8 objective mines the top five in-batch negatives using the frozen DQU
query. It requires the structured residual to improve the positive-versus-hard
negative gap over the frozen DQU gap by a margin of 0.01. The DQU text branch,
image branch, targets, structured gate, and negative selection are detached in
this term, so its gradients can update only the bounded structured adapter
residual.

| Configuration (seed 42) | Best epoch | R@1 | R@10 | R@50 | R@10+R@50 | Mean rank |
|---|---:|---:|---:|---:|---:|---:|
| v7, no margin loss | 6 | 21.6163 | 52.7516 | 75.1116 | **127.8632** | 98.3520 |
| v8, weight 0.20, k=5, margin=0.01 | 6 | 21.6658 | 52.6525 | 75.1116 | 127.7640 | 98.3818 |
| v8, weight 0.05, k=5, margin=0.01 | 6 | 21.6658 | 52.6525 | 75.1116 | 127.7640 | 98.3767 |

Both v8 runs stopped at epoch 11. In the weight-0.20 run, the margin violation
rate fell from 0.9880 at the first adapter epoch to 0.7441 at early stopping.
For weight 0.05 it fell from 0.9910 to 0.7570. The loss therefore learns the
intended residual ranking constraint, but neither weight improves the
selection metric over v7. Relative to v7 seed 42, v8 trades +0.0496 R@1 for
-0.0992 R@10, with unchanged R@50. Lowering only the loss weight does not
change that trade-off.

## Structured keyword-written image route

The official DQU image route writes extracted target keywords onto the
reference image. The structured variant writes only Qwen `add` attributes:
`remove` attributes are deliberately excluded because rendering an unwanted
concept may strengthen it in CLIP image space. The original DQU rendering is
kept byte-for-byte as the control.

Frozen-DQU, zero-training seed-42 ablations:

| Image input | R@1 | R@10 | R@50 | R@10+R@50 | Mean rank |
|---|---:|---:|---:|---:|---:|
| Original DQU-written image | 21.6658 | 52.1567 | 74.9132 | 127.0699 | 99.3713 |
| Qwen-add replaces DQU keywords | 21.5171 | 52.0575 | 75.1611 | 127.2186 | 98.5930 |
| Qwen-add and DQU words on one image | 21.3684 | 51.7105 | 75.0124 | 126.7229 | 98.7412 |
| Feature blend, Qwen alpha=0.6 | 21.8642 | 52.4541 | 75.4090 | **127.8632** | 98.5007 |

Pixel-level word concatenation hurts, while separately encoding the two written
images and blending their normalized CLIP image features is complementary.
The image-only alpha sweep reproduces DQU exactly at alpha=0 and Qwen
replacement exactly at alpha=1.

The best combination uses the v7 structured-text adapter and blends complete
DQU-written and Qwen-written query views with fixed alpha=0.5. Alpha was chosen
on seed 42, then held fixed for seeds 43 and 44.

| Seed | R@1 | R@10 | R@50 | R@10+R@50 | Mean rank | Delta score vs v7 |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 21.9633 | 52.9995 | 75.4090 | 128.4085 | 97.3708 | +0.5454 |
| 43 | 21.9633 | 52.7020 | 75.2603 | 127.9623 | 97.9380 | +0.3966 |
| 44 | 22.1120 | 52.7516 | 75.1611 | 127.9127 | 97.8389 | +0.5454 |

| Metric | Three-seed mean | Sample standard deviation | v7 mean | Mean delta |
|---|---:|---:|---:|---:|
| R@1 | 22.0129 | 0.0859 | 21.7154 | +0.2975 |
| R@10 | 52.8177 | 0.1594 | 52.5368 | +0.2809 |
| R@50 | 75.2768 | 0.1248 | 75.0620 | +0.2148 |
| R@10+R@50 | 128.0945 | 0.2731 | 127.5987 | +0.4958 |
| Mean rank | 97.7159 | 0.3029 | 98.6882 | -0.9722 |

All three seeds improve every reported recall metric in the mean. FashionIQ
does not provide a separate public validation set for this tuning protocol, so
alpha=0.5 should remain fixed in subsequent category experiments rather than
being retuned on each evaluation set.

## Conclusions and next priorities

1. Structured JSON is useful as a small, protected correction to DQU, not as a
   replacement for the natural-language modification.
2. The dual written-image view is the strongest tested next step. It improves
   all three recalls and mean rank across the three v7 seeds, without changing
   DQU weights or retraining CLIP.
3. Confidence should remain soft. A hard 0.9 threshold would discard the
   lower-confidence group's useful R@50 and mean-rank gains.
4. Do not concatenate both keyword sets on one image. Preserve two visual
   views and fuse their features or final normalized queries.
5. The next model change should replace the fixed alpha with a bounded,
   confidence-aware image residual gate initialized to the fixed 0.5 solution.
   It should be regularized toward DQU and evaluated on shirt/toptee with alpha
   fixed before any further dress tuning.
