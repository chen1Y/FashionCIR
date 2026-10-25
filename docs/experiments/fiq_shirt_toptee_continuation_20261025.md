# Shirt / Toptee continuation on the cloned server

Date: 2026-10-25, Hong Kong. New server: `connect.westc.seetacloud.com:33699`.
The user explicitly requested completion of the paused original-DQU versus v7 off/identity/annotated comparisons, and a remaining-work report **before launching experiments**. That report was provided after clone verification and before generation/training launch.

## Verified saved work and remaining scope

The completed Shirt baseline remains R@1/10/50=24.7791952895/54.5632973503/73.2580961727, best epoch2 after seven training epochs. Its full checkpoint SHA-256 is unchanged: `6690822bf59dbc4c6c9e45df9d2a0026da033b327afef595bbc17829d2150866`. It is adopted, never retrained.

| Online split | Already finalized | Remaining |
|---|---:|---:|
| Shirt train | 2677 (2521 program-valid) | 3311 |
| Shirt val | 0 | 2038 |
| Toptee train | 0 | 6027 |
| Toptee val | 0 | 1961 |
| Total | 2677 | **13337** |

Seven GPU stages remain: Shirt structured off/identity/annotated, Toptee original DQU, then Toptee structured off/identity/annotated. Each of the six adapter runs has six epochs; Toptee baseline retains max100 epochs/patience5, with actual stopping/best epochs reported. Full original-split galleries and validation queries remain unchanged. No data expansion, additional seeds, dynamic fusion, re-ranking, image generation or other suggestions from the external AI review are introduced.

Original saved commands, source/model/metadata hashes, dataset counts, ledger identity and the completed baseline are validated before starting. Three new CPU guard tests check completed-stage adoption, unchanged remaining commands, and no runtime reset. Old script/model/generator source remains byte-identical. Only `scripts/resume_fiq_saved_pause.py` is added to adopt the completed stage and execute the exact remaining commands from `pause_resume_plan.json`.

## Runtime and cost continuity

The user-requested pause between October22 and this explicit continuation is excluded from active queue time. **No consumed runtime is reset:** original43200 seconds minus3152.034592 already spent leaves40047.965408 seconds (about11h7m28s) for this continuation. `continuation_budget_20261025.json` records the new launch time and this remaining ceiling; an existing continuation file cannot be overwritten to obtain a fresh budget.

The same SQLite ledger is reused, including2677 successful/failed finalized samples,2910 known requests and the single old unknown request. Query157 stays a saved failure/DQU fallback; it is not reissued. Old measured list-price estimate1.6507186 CNY plus0.004096 CNY unknown allowance gives1.6548146 CNY conservative accounting, leaving13.3451854 CNY under the original15 CNY total cap. Each category retains its7.5 CNY cap; no spending limit is replenished by migration.

The unchanged online generator uses Qwen3.7-Flash, original Dress prompts/rules, reference image plus modification only, eight workers/initial120RPM, at most one repair of a known invalid response, and no automatic network/unknown retries. All queries must be finalized with at least90% program-valid coverage per split before a category's structured training starts. These checks are not human semantic ground truth.

Beijing <=32k list prices rechecked on2026-10-25 remain input0.2/output0.8 CNY per million tokens. Actual token usage and conservative reservations are recorded per category/split; estimates are not invoices. [Official prices](https://help.aliyun.com/zh/model-studio/qwen3-7-flash).

## Resources and monitoring

The clone has an idle RTX4080 SUPER with32760MiB reported memory,12.419GiB free data disk and31GiB free tmpfs at initial verification. Full Shirt checkpoint, original sources and existing ledger match the paused instance. Predicted persistent room for the additional Toptee baseline plus0.5GiB generation/log allowance is checked while preserving8GiB. Toptee training bests use the original dedicated tmpfs staging, followed by byte-identical SHA-verified durable publication. No valid older weights are deleted.

The code stays in `/root/autodl-tmp/experiments/fiq-shirt-toptee-online-20261022`; results stay in `/root/autodl-tmp/results/fiq_shirt_toptee_20261022`. Previous pause reports and current PID records are backed up to a timestamped `continuation_*` directory. Logs append explicit continuation markers; the original pause's KeyboardInterrupt is historical evidence, not a new failure. `resume_preflight_20261025.json` records initial log offsets for monitoring.

Generation resumes first; GPU may remain idle while waiting for complete Shirt train/val. GPU stages remain sequential in the original order. The dispatcher stops on API failure, insufficient coverage/budget, OOM/non-finite loss, disk<8GiB or exhausted remaining time. No stage is silently repeated or auto-resumed after failure. Final per-category comparisons require eight total completed GPU stages (including the adopted Shirt baseline) and complete valid generation outputs. Intermediate bests remain labeled intermediate; no significance claim from a single seed.

## Verified launch

All three new guard tests passed locally and on the new server. The detached continuation launched at2026-10-25T15:43:14.106045+00:00 (23:43 Hong Kong). Initial dispatcher PID2098 and online child2099 were verified by command/cwd; these are historical references only. New API responses returned normally and finalized records advanced from2677 to2958 at the evidence snapshot. Old query157 still has exactly its original unknown attempt. No Shirt baseline training was launched; GPU is correctly idle while waiting for complete online Shirt inputs.

Launch evidence is saved under `results/fiq_shirt_toptee_20261022/continuation_20261025`. Server files remain at the original result root for the dispatcher. `experiment_report.json.status=waiting_online_shirt` is intermediate, not a final comparison. Automated monitoring has been reactivated for the new server only. Final seven-stage outcomes and total costs are pending.
