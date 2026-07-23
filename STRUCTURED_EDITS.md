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
