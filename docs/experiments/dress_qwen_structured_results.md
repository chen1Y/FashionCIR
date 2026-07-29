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

## Conclusions and next priorities

1. Structured JSON is useful as a small, protected correction to DQU, not as a
   replacement for the natural-language modification.
2. The reproducible gain is concentrated in R@10/R@50 and mean rank. R@1 is
   essentially unchanged, so the next loss should explicitly target hard
   top-ranked negatives.
3. Confidence should remain soft. A hard 0.9 threshold would discard the
   lower-confidence group's useful R@50 and mean-rank gains.
4. The next highest-value model experiment is a top-k hard-negative or
   margin-ranking objective applied only to the bounded structured residual.
5. After that, test field dropout during adapter training to reduce dependence
   on Qwen field completeness. More complex learned field attention is lower
   priority unless it is regularized and proven stable.
