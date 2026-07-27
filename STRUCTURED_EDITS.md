# Qwen3-VL structured edit generation

`generate_structured_edits.py` converts each FashionIQ query into a validated
edit program. Qwen receives only:

- the reference image;
- the original modification text.

The target image and target identifier are never placed in the model prompt.
They remain in the output only as evaluation metadata.

## 1. Inspect inputs without loading Qwen

Run from `src/`:

```powershell
python generate_structured_edits.py `
  --fashioniq-root ../data/FashionIQ `
  --category dress `
  --split train `
  --limit 3 `
  --dry-run
```

Check that the printed `messages` contain no `target` field.

## 2. Small feasibility run

```powershell
python generate_structured_edits.py `
  --fashioniq-root ../data/FashionIQ `
  --category dress `
  --split train `
  --model Qwen/Qwen3-VL-8B-Instruct `
  --limit 20
```

If the model already exists in the local Hugging Face cache, add `--offline`.
Use `--attn-implementation flash_attention_2` only when FlashAttention is
installed; the default `sdpa` is more portable.

Default output:

```text
../data/FashionIQ/captions/structured_edits_dress_train.json
```

The output is checkpointed every five samples and can be resumed by rerunning
the same command. Successfully validated samples are skipped.

## 3. Validate an existing output

```powershell
python generate_structured_edits.py `
  --fashioniq-root ../data/FashionIQ `
  --category dress `
  --split train `
  --validate-only
```

Validation reports schema errors and semantic errors separately. Semantic checks
cover duplicate attributes, cross-field conflicts, irrelevant person/background
details, incompatible target properties, and removed attributes that remain in
the target description.

## 4. Sanitize and repair an existing output

Back up the JSON first, then run:

```powershell
python generate_structured_edits.py `
  --fashioniq-root ../data/FashionIQ `
  --category dress `
  --split train `
  --model Qwen/Qwen3-VL-8B-Instruct `
  --repair-existing `
  --limit 0
```

All selected samples receive deterministic deduplication and cleanup. Qwen is
loaded only when semantic errors remain after cleanup, and receives only the
reference image, modification text, previous JSON, and validation errors. The
target image and target identifier remain excluded.

## 5. Recommended feasibility evaluation

Generate 50-100 samples spanning dresses, shirts, and tops/tees. Manually score:

1. whether `remove` captures the source property being replaced;
2. whether `add` captures every requested target property;
3. whether negation and comparisons are preserved;
4. whether `target_description` is consistent with the edit program;
5. whether any unsupported visual attribute is hallucinated.

Do not use target images during this review. They may be inspected only in a
separate downstream retrieval evaluation.

## 6. Frozen-CLIP A/B retrieval diagnostic

`evaluate_structured_edits.py` compares several representations while keeping
the OpenCLIP encoder, gallery, queries, and fusion weights fixed:

- the original FashionIQ modification text;
- Qwen's short `target_description`;
- a serialized structured template;
- separately encoded structured fields;
- raw/target text-feature hybrids;
- target-description residuals added to the original image-text query.

Example using a full FashionIQ validation gallery:

```bash
python evaluate_structured_edits.py \
  --fashioniq-root ../data/FashionIQ \
  --structured-json ../data/FashionIQ/captions/structured_edits_dress_val_50.json \
  --category dress \
  --split val \
  --gallery split \
  --model ViT-B-32 \
  --pretrained openai \
  --cache-dir ../model_cache \
  --image-weights 0.25,0.5,0.75 \
  --target-weights 0.25,0.5,0.75 \
  --bootstrap-samples 5000
```

The output contains R@1/R@10/R@50, mean and median target rank, paired
bootstrap confidence intervals against the matching raw-text baseline, and
per-query ranks. Choose fusion weights on a training/development subset and
report the corresponding validation result; selecting the best validation
weight would bias the estimate.

This script is a frozen-encoder diagnostic rather than a replacement for the
project's trained retrieval evaluation. A positive result should subsequently
be verified with the same trained checkpoint and the official full validation
split.

## 7. Trainable structured-description gate

`newTrain.py` now uses `newDataset.FashionIQ`, `newModel.DQU_CIR`, and
`newTest.test` as one consistent path. Qwen is not loaded during retrieval
training. The model encodes:

1. the original FashionIQ modification;
2. the offline Qwen `target_description`;
3. the unmodified reference image.

A learned text gate mixes (1) and (2), then a learned image gate mixes the
result with (3). Both gates start at 0.25, based on the frozen-encoder
diagnostic. Samples without a validated structured description automatically
fall back to the original text.

Example using a local ViT-H/14 checkpoint:

```bash
python newTrain.py \
  --fashioniq_path ../data/FashionIQ \
  --dataset dress \
  --fashioniq_split original-split \
  --structured-train-path ../data/FashionIQ/captions/structured_edits_dress_train.json \
  --structured-val-path ../data/FashionIQ/captions/structured_edits_dress_val.json \
  --clip-model ViT-H-14 \
  --clip-checkpoint /path/to/open_clip_pytorch_model.bin \
  --batch-size 16 \
  --num-epochs 20
```

Use `--structured-only` for a small feasibility subset. Do not report that
subset as the official FashionIQ result. Without this flag, missing structured
descriptions fall back safely to the original text and the logged coverage
shows how much data actually trains the structured branch.

Useful checks:

```bash
# Evaluate the structured branch without training.
python newTrain.py ... --structured-only --eval-only

# Exact same evaluation with the target-description branch disabled.
python newTrain.py ... --structured-only --eval-only --disable-target-description

# Reproduce the text-written-on-image raw-data baseline explicitly.
python newTrain.py ... --use-written-image
```

`newTrain.py` intentionally imports `newTest`, not the legacy `test.py`.
Reported runs should use `original-split`, which ranks against the full
FashionIQ validation gallery and removes the reference image before computing
R@1, R@10, and R@50.
