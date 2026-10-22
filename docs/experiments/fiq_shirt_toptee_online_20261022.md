# Shirt / Toptee original-DQU and structured comparisons

Date: 2026-10-22, Hong Kong. Server: `connect.weste.seetacloud.com:47914`.
Branch: `codex/fiq-shirt-toptee-online-20261022`.

The user clarified that this run completes **original DQU versus v7 structured off/identity/annotated**, without augmenting either dataset. It does not repeat the Dress augmentation experiment.

## Data and generation

| Category | Train queries | Validation queries | Gallery |
|---|---:|---:|---:|
| Shirt | 5988 | 2038 | 6346 |
| Toptee | 6027 | 1961 | 5373 |

Full original training data and validation galleries are retained; reference images are excluded in retrieval. No test labels select models. Dataset metadata hashes are frozen in `protocol.json`. No Dress checkpoint substitutes for a Shirt/Toptee baseline.

Online Qwen3.7-Flash uses the existing FashionIQ Dress `SYSTEM_PROMPT`, `build_user_prompt`, schema checks, sanitation and deterministic semantic checks. Only the reference image and original modification are transmitted. Target IDs are output binding metadata; target images/descriptions never enter the prompt. This is not an independent-model/human quality judgment. Every query receives a final saved success/failure record; failures explicitly fall back to DQU.

All four full splits total **16014 queries**. At most one repair is permitted after a known returned invalid JSON/program; no paid network/unknown-request retries. Maximum 32028 calls, **15 CNY total** (7.5 CNY per category), 8 workers, initially 120 RPM, 1024 output-token limit. HTTP429 reduces rate; repeated errors and authentication errors stop the run. Final full split coverage must reach 90% before structured training.

Durable SQLite ledgers are separate by category and shared across train/val. Known responses and finalized failures are reusable; unknown reservations are never blindly reissued and budgets do not reset on a resumed ledger. Cost summaries report calls, prompt/completion tokens, measured list-price estimates, conservative unmeasured reservations and per-category/per-split totals. Beijing official <=32k price checked on 2026-10-22: input 0.2 CNY/Mtoken and output 0.8 CNY/Mtoken; discounts are not assumed and estimates are not invoices. [Official pricing](https://help.aliyun.com/zh/model-studio/qwen3-7-flash).

The API credential is entered with a hidden prompt, inherited in dispatcher memory and passed through an anonymous pipe to generation. It is never persisted in argv, environment files, logs, source or Git. Core dumps are disabled.

## Training protocol

GPU work is sequential: Shirt DQU -> Shirt off/identity/annotated -> Toptee DQU -> Toptee off/identity/annotated. API generation runs concurrently on CPU/network, processing Shirt then Toptee. Each category requires its own trained baseline.

- Baseline matches previous Dress DQU: ViT-H/14, original captions/keywords and transforms, seed42, batch16, CLIP lr1e-6, fusion lr1e-4, FP16 AMP, at most100 epochs, early stopping after5 stale validation scores, score=R@10+R@50. Actual stopped/best epochs are reported; no training truncation.
- Each v7 branch freezes the same category baseline, seed42, batch16, lr5e-5, 2 gate +4 adapter epochs, BF16 train/FP16 eval, hard-negative weight0. Only the positive-label mode changes. Identity uses identical target IDs; annotated additionally uses exact same-reference/same-modification training labels. Semantically similar images are not automatically positives. Sparse label coverage and identical branch outcomes are reported honestly.
- Every branch evaluates the entire unchanged validation set/gallery and chooses one checkpoint by R@10+R@50. All R@k come from that checkpoint. Partial bests are intermediate; missing values are pending, not zero. Final comparison needs eight successful GPU stages and four complete generation splits.

These are single-seed comparisons of the residual/multi-positive method with a fixed trained DQU per category, not equal-compute end-to-end model retraining or proof of significance. Dress results from earlier runs remain separately labeled; no mixed-protocol three-category average is presented without checking compatibility.

## Resource protection and reproducibility

The clone initially had 15.025 GiB free and no Shirt/Toptee baselines. Only inactive `.incomplete` download fragments were cleared after checking all process file descriptors; `storage_cleanup.json` lists the18 files and allocated bytes. Complete models, previous checkpoints and evidence were preserved. About1.188 GiB was recovered, leaving16.213 GiB.

New full best checkpoints are about3.69 GiB each. To avoid persistent double-copy peaks during every best replacement, baseline bests are written atomically in a dedicated `/dev/shm/fiq_shirt_toptee_20261022/<category>` directory. Once the baseline finishes successfully, exactly one unchanged full checkpoint is copied to data-disk `.partial`, fsynced, SHA-verified and atomically published; the verified staging copy is then removed. **No dtype conversion, tensor dropping or checkpoint approximation** is used. Staging bests are volatile until publication and are never labeled durable/completed beforehand.

The dispatcher verifies projected room for both durable baselines, checks the data disk every10 seconds and before publication, preserves at least8 GiB, checks tmpfs room, and stops on OOM/non-finite losses/stage failure. A12-hour total queue wall limit is fixed at launch, including online generation; no implicit restart, new seed, expansion, CIG generation or extra training is queued. Small adapter-only checkpoints depend on their preserved original baselines.

Code: `src/generate_fashioniq_online.py`, `scripts/run_fiq_shirt_toptee.py`.
Server code: `/root/autodl-tmp/experiments/fiq-shirt-toptee-online-20261022`.
Results: `/root/autodl-tmp/results/fiq_shirt_toptee_20261022`.
Read `experiment_report.json`, `protocol.json`, `cost_summary.json`, per-category ledgers/structured JSON/logs/checkpoints and current PIDs with `/proc` identity checks. Clone-inherited PID numbers are never trusted.

## Launch verification

Three query-binding/publication guard tests passed locally and on the server; ten existing durable-ledger/cost/resume tests passed on the server. Both real FashionIQ adapters loaded all original metadata, checked full validation/gallery sizes and actually drew/read a training pair per category. Exact annotated multi-positive coverage is96/5988 queries for Shirt and38/6027 for Toptee; these are conservative existing training annotations, not newly generated positives.

The detached dispatcher launched on2026-10-22: initial queue PID3553, online generation3554 and Shirt DQU3555, with commands/cwd/GPU identity verified. These PIDs are only historical references; monitor current `/proc` identity. Online generation and Shirt baseline initialization are active. Final metrics and actual cost remain pending; no retrieval benefit is claimed before completion.


## Explicit continuation after account replenishment

The initial API process stopped on one HTTP403 at Shirt train query157, after159 finalized query records and169 known requests. HTTP403's exact provider cause was not returned by the old client; the user subsequently replenished the Qwen account and explicitly requested continuation. No claim that it was a content-safety rejection is made.

Initial measured list-price usage was0.0966416 CNY. The single unknown request retains0.004096 CNY conservative reservation, giving0.1007376 CNY total upper accounting at the stop. Its saved failed query remains a DQU fallback and is not retried. All159 records (including failures) and the original SQLite budget are reused.

The interrupted Shirt baseline had reached182/375 batches of epoch1 but had no checkpoint or saved optimizer state. Its complete same-seed baseline must therefore start again; the interrupted work is documented as overhead and is not presented as resumed model steps. Original queue/generation/training logs, protocol, reports and partial generated output are copied to a timestamped recovery directory, and initial source files are retained separately. Only explicit initial-failure dispatch recovery code changes; generation, prompts, schemas, training, data, per-category7.5 CNY/total15 CNY limits and the original12-hour launch deadline stay fixed. The dispatcher refuses a recovery that changes other protocol/code fields or has an existing training checkpoint. An additional guard test verifies these restrictions.


Continuation was verified on the new dispatcher3958, online process3959 and Shirt DQU3960 (historical references only): new API requests returned normally, total finalized samples advanced beyond159, and same-seed Shirt baseline epoch1 batches advanced. Original start remains2026-10-22T13:56:44.261471+00:00; recovery evidence is in `recovery_verification.json`. No new retrieval result is available yet. Monitoring remains active with the original spending/time limits.
