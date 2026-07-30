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

## Local generative-edit view pilot

Date: 2026-07-30. A leakage-free pilot tested whether a generated image view
can improve the best seed-42 dual written-image query. The generator receives
only the reference image and the Qwen structured edit; the target image is
never opened. `stable-diffusion-v1-5/stable-diffusion-inpainting` locally
repaints a coarse region derived from the structured `region` fields. The
generated complete-query residual is:

`q = normalize(q_dual + lambda * confidence * (q_generated - q_plain))`

The pilot uses 120 validation queries selected round-robin by the primary
structured edit category. The retrieval gallery and all 2,017 validation
queries remain unchanged. A fixed random permutation of generated residuals is
the negative control.

### V1: initial regional masks

Visual QA found that upper/full masks could modify a model's face. This version
is therefore diagnostic only.

| lambda | R@1 | R@10 | R@50 | R@10+R@50 | Mean rank | Selected wins/ties/losses |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 21.9633 | 52.9995 | 75.4090 | **128.4085** | 97.3708 | 0/120/0 |
| 0.05 | 21.9633 | 52.9995 | 75.4090 | **128.4085** | 97.3738 | 23/76/21 |
| 0.10 | 21.9633 | 52.9499 | 75.4090 | 128.3589 | **97.3694** | 28/68/24 |
| 0.15 | 21.9633 | 52.9499 | 75.3594 | 128.3094 | 97.3823 | 32/61/27 |
| 0.20 | 21.9633 | 52.9499 | 75.3099 | 128.2598 | 97.3798 | 34/55/31 |

The true residual is less damaging to mean rank than the shuffled residual
(for example, +0.15 versus +3.02 selected mean-rank change at lambda 0.20),
which indicates weak semantic signal. It does not improve recall.

### V2: face-protected masks and invalid-image guard

The upper mask begins at normalized y=0.20 and the full mask at y=0.18, keeping
faces and hair outside the generated region. Black or near-constant safety
outputs are treated as missing views and contribute zero residual. Five of 120
outputs were excluded by this guard; one was explicitly reported by the
Diffusers safety checker.

| lambda | R@1 | R@10 | R@50 | R@10+R@50 | Mean rank | Selected wins/ties/losses |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 21.9633 | 52.9995 | 75.4090 | **128.4085** | **97.3708** | 0/115/0 |
| 0.05 | 21.9633 | 52.9995 | 75.4090 | **128.4085** | 97.3743 | 21/76/18 |
| 0.10 | 21.9633 | 52.9499 | 75.4090 | 128.3589 | 97.3783 | 25/65/25 |
| 0.15 | 21.9633 | 52.9499 | 75.4090 | 128.3589 | 97.3798 | 25/65/25 |
| 0.20 | 21.9633 | 52.9499 | 75.4090 | 128.3589 | 97.3773 | 27/59/29 |

### Decision

Do not expand this SD1.5 coarse-inpainting route to the complete validation or
training set. It changes visible garment details but does not beat the frozen
dual-view baseline, and mask cleanup alone does not recover a gain. Keep the
integration and controls for future generators. A stronger reference-
conditioned editor should first pass the same 120-query protocol, including
the shuffled-residual control, before any full-data generation. The most
useful next variant is a garment-segmentation mask plus an appearance-
preserving editor, with its reliability gate trained on train only and all
dress validation hyperparameters fixed.

## Qwen-Image-Edit generated-view pilot

Date: 2026-07-30. The official base `Qwen/Qwen-Image-Edit` checkpoint was
tested as a stronger reference-conditioned editor. The checkpoint was obtained
from its official ModelScope mirror because the Hugging Face transfer repeatedly
timed out. The target image is still used only as the retrieval label and is
never passed to the generator.

The final pilot configuration was NF4 transformer/text-encoder quantization,
512 x 512 output, 20 inference steps, batch size 1, and global semantic editing
followed by compositing the source pixels back outside the structured garment
region. All 20 deterministically selected validation queries generated a valid
file. This is an intentionally small stopping-gate run, not a result on the
complete validation split.

### Deployment and smoke-test observations

| Configuration | Approx. time/image | Peak/runtime GPU memory | Observation |
|---|---:|---:|---|
| NF4, 512, 20 steps, batch 1 | 51-62 s | about 16.1-16.5 GiB | Best tested semantic compliance; severe mosaic/watercolor artifacts remained |
| NF4, 512, 10 steps, batch 1 | about 33 s | about 16 GiB | Faster, but artifacts and missing details increased |
| NF4, 512, 10 steps, batch 2/4 | about 29/28 s | up to about 17.5 GiB | Little throughput gain; batch 4 changed outputs and visibly reduced quality |
| NF4, 768, 10 steps, batch 1 | about 37 s | about 16 GiB | Sharper, but some requested geometry edits were missed |
| INT8, 512, 20 steps, batch 1 | about 102 s | about 30.4 GiB runtime | Cleaner, but slower and still missed requested geometry edits |

For example, the global editor could change a red long gown to a black short
gown, but it also changed body pose and introduced blocky texture. Inpainting
preserved more of the source, but frequently failed to perform the requested
length change. Therefore generation validity alone is not treated as evidence
that the view is useful for retrieval.

### Retrieval protocol

The frozen best seed-42 dual written-image query remains the baseline:
R@1=21.9633, R@10=52.9995, R@50=75.4090, R@10+R@50=128.4085, and mean
rank=97.3708. The 20 selected queries have baseline R@1=20, R@10=45,
R@50=65, and mean rank=112.65.

Two fusion rules were evaluated:

- `residual`: add the confidence-weighted difference between the generated
  complete-query view and the plain DQU complete-query view;
- `direct`: interpolate the confidence-weighted generated complete-query view
  as a third query view.

For each rule, the exact generated view is compared with a fixed shuffled-view
negative control. A lower selected mean-rank delta is better.

| Fusion | lambda | Matched selected mean-rank delta | Shuffled delta | Matched full score |
|---|---:|---:|---:|---:|
| residual | 0.025 | +0.85 | -0.95 | 128.4085 |
| residual | 0.050 | +1.45 | -1.60 | 128.4085 |
| residual | 0.100 | +3.40 | -3.35 | 128.4085 |
| residual | 0.150 | +4.60 | -4.00 | 128.4085 |
| residual | 0.200 | +8.05 | -5.00 | 128.3589 |
| direct | 0.025 | +0.50 | -1.55 | 128.4085 |
| direct | 0.050 | +1.10 | -4.50 | 128.4085 |
| direct | 0.100 | +1.85 | -7.40 | 128.4085 |
| direct | 0.150 | +2.95 | -8.90 | 128.4085 |
| direct | 0.200 | +3.80 | -7.50 | 128.4085 |

The matched generated views do not improve R@10 or R@50 at any tested direct
weight. At residual lambda=0.20, full-split R@50 decreases from 75.4090 to
75.3594. The small full-score increases seen for shuffled direct views
(128.4581 at lambda=0.10 and 128.5077 at lambda=0.15/0.20) are explicitly not
model gains: they come from unrelated generated images and therefore expose
random or nonspecific feature perturbation.

### Decision

Do not expand this base Qwen-Image-Edit NF4 route from 20 to 120 or to the full
split, and do not add these generated views to `newModel.py`. Both matched
fusion rules fail the shuffled negative control, while visual inspection shows
loss of identity, garment detail, and texture. The experiment validates the
pipeline and establishes a reusable early stopping protocol, but it rejects
the current generator/configuration as a retrieval improvement.

If image generation is revisited, first test a higher-fidelity editor such as
Qwen-Image-Edit-2511, or a reliable pre-quantized higher-precision checkpoint,
on these same 20 queries with fixed weights and the same shuffled control.
Only a configuration whose matched view beats both the frozen baseline and the
shuffled control should proceed to 120 queries. A learned reliability gate
must be trained on FashionIQ train only rather than selected on validation.

## Qwen mask ablation: none versus fixed versus CLIPSeg

Date: 2026-07-30. The fixed rectangular composite mask is not appropriate for
every FashionIQ image. A controlled 20-query ablation therefore compares:

- `none`: use the complete Qwen output without compositing;
- `fixed`: composite with the existing region-dependent rounded rectangle;
- `clipseg`: segment the garment from the source with
  `CIDAS/clipseg-rd64-refined`, dilate the mask by 4% of image width to permit
  shape changes, feather its edge, and composite only that region.

All three routes use the same raw Qwen image for every query. Thus their
differences are caused only by masking, not by random generation. The revised
Qwen prompt says to edit the garment rather than referring to a mask that the
global editor does not receive.

Nineteen of 20 CLIPSeg masks were accepted. Before dilation, accepted garment
coverage ranged from 11.28% to 43.70% (mean 23.87%). One empty segmentation
fell back to the fixed mask. Visual inspection shows that CLIPSeg follows the
dress silhouette and protects faces/backgrounds much better than a fixed
rectangle. It cannot remove mosaic artifacts within the generated garment.

The table reports selected-query mean-rank change; lower is better. Every
matched route is also compared with its shuffled generated-view control.

| Fusion | Weight | No mask matched/shuffled | Fixed matched/shuffled | CLIPSeg matched/shuffled |
|---|---:|---:|---:|---:|
| residual | 0.025 | **+0.15 / -0.70** | +1.30 / -0.65 | +0.80 / 0.00 |
| residual | 0.050 | **+0.85 / -1.25** | +2.45 / -1.20 | +2.00 / +0.05 |
| residual | 0.100 | **+1.65 / -1.85** | +4.85 / -2.20 | +4.25 / +0.90 |
| residual | 0.200 | **+7.30 / -0.95** | +12.85 / -1.85 | +14.10 / +5.85 |
| direct | 0.025 | **+0.20 / -1.60** | +0.70 / -1.85 | +0.65 / -1.20 |
| direct | 0.050 | **+0.40 / -3.55** | +2.05 / -3.75 | +1.75 / -2.00 |
| direct | 0.100 | **+1.05 / -5.85** | +3.70 / -5.35 | +2.90 / -2.65 |
| direct | 0.200 | **+2.45 / -6.35** | +7.55 / -2.75 | +8.00 / +1.25 |

No matched configuration improves full-split R@10 or R@50. Their score remains
128.4085, except CLIPSeg residual at weight 0.20, which reduces R@50 from
75.4090 to 75.3594 and score to 128.3589. The unmasked route is consistently
the least damaging, but even its smallest tested weight worsens selected mean
rank and loses to the shuffled control.

### Mask-ablation decision

The fixed-mask concern is valid, and CLIPSeg is a clear visual improvement over
the rectangular mask. However, masking is not the primary cause of the failed
retrieval route. The underlying NF4 Qwen output changes pose, scale, texture,
and background too strongly. Removing the mask avoids hard composite seams
and performs best of the three, but still supplies no useful matched semantic
signal.

Do not add any of these three generated views to `newModel.py` and do not scale
this base Qwen configuration beyond the 20-query stopping gate. Keep CLIPSeg as
the preferred mask implementation for a future higher-fidelity editor, because
it solves the geometric mask mismatch even though it cannot rescue the current
generator. A future editor must first beat both the unmasked result and the
shuffled control on this exact sample set.

## Qwen-Image text-to-image target hypotheses

Date: 2026-07-30. This route removes image editing entirely. The official
`Qwen/Qwen-Image` text-to-image model receives only the Qwen structured
`target_description` and added attributes. Neither the FashionIQ reference
image nor target image is opened by the generator. The output is treated as a
standalone visual hypothesis of the target rather than as a new reference image
to which the relative caption is applied again.

To fit the 150 GiB data disk, only the Qwen-Image generation transformer was
downloaded. Its text encoder, tokenizer, and VAE are shared by symbolic links
with the existing Qwen-Image-Edit checkpoint. The added transformer occupies
about 39 GiB. Generation uses NF4 quantization, 512 x 512 resolution, 20 steps,
and a fixed ecommerce-catalog prompt. It takes about 10 seconds per image and
uses about 16.1-17.0 GiB GPU memory, substantially faster than the base
Qwen-Image-Edit experiment.

The generated target feature is fused only by direct interpolation:

`q = normalize((1 - lambda * confidence) * q_dual + lambda * confidence * g)`

where `g` is the normalized CLIP image feature of the text-generated target
hypothesis. A fixed shuffled assignment of generated hypotheses is the negative
control.

### 20-query stopping gate

The first 20 queries showed a small high-weight recall signal, but the selected
mean rank degraded. At lambda=0.40, R@1 increased from 21.9633 to 22.0625 and
R@50 from 75.4090 to 75.4586, while selected mean-rank change was +13.55.
This was insufficient by itself, but generation was fast enough to justify the
predefined 120-query confirmation.

### 120-query confirmation

The selected-query baseline is R@1=18.3333, R@10=51.6667, R@50=68.3333,
and mean rank=125.2167.

| lambda | R@1 | R@10 | R@50 | R@10+R@50 | Full mean rank | Selected rank delta | Shuffled rank delta |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.000 | 21.9633 | 52.9995 | 75.4090 | 128.4085 | 97.3708 | 0.000 | 0.000 |
| 0.025 | **22.0129** | 52.9995 | 75.4090 | 128.4085 | 97.3426 | -0.475 | -0.050 |
| 0.050 | 21.9633 | 52.9499 | 75.4090 | 128.3589 | 97.3223 | -0.817 | +0.200 |
| 0.075 | **22.0129** | 52.9499 | 75.4090 | 128.3589 | 97.2876 | -1.400 | +0.808 |
| 0.100 | 21.9137 | **52.9995** | **75.5082** | **128.5077** | **97.2623** | **-1.825** | +1.383 |
| 0.150 | 21.9137 | 52.9003 | 75.5082 | 128.4085 | 97.2568 | -1.917 | +2.492 |
| 0.200 | 21.8642 | 52.8508 | 75.5578 | 128.4085 | 97.2543 | -1.958 | +4.442 |
| 0.300 | 21.9137 | 52.8508 | 75.5578 | 128.4085 | 97.3089 | -1.042 | +11.633 |
| 0.400 | 21.9633 | 52.8012 | 75.5082 | 128.3094 | 97.4799 | +1.833 | +25.217 |
| 0.500 | 21.8642 | 52.7020 | 75.5082 | 128.2102 | 97.8820 | +8.592 | +51.408 |

At lambda=0.10, two of the 120 selected queries cross into R@50 while one
crosses out of R@1. Full mean rank improves by 0.1086 and selected mean rank by
1.825. The shuffled control has score 128.4085 and worsens selected mean rank
by 1.383, so the matched result is not explained by generic generated-image
noise. Unlike the image-edit route, this route passes the semantic negative
control at its best score.

### Text-generation decision

Text-to-image target hypotheses are a positive but small improvement:
R@10+R@50 rises by 0.0992 to 128.5077. This is the first generated-image route
in these experiments that improves the primary score, overall mean rank, and
the shuffled-control comparison simultaneously.

Do not yet claim a final model improvement or tune lambda further on dress
validation. Fix lambda=0.10 and evaluate shirt/toptee next, or train a bounded
reliability gate on FashionIQ train only. The current NF4 images still contain
face and fine-texture artifacts, so a higher-fidelity generator or an
image-quality/semantic-consistency gate may strengthen the signal.
