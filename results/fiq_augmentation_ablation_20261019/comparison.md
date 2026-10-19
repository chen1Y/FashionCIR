# FashionIQ Dress augmentation: final matched-budget comparison

| Arm | Unique training triplets | R@1 | R@10 | R@50 |
|---|---:|---:|---:|---:|
| original_padded | 5985 | 21.6658 | 52.6525 | 75.2603 |
| augmented101 | 6086 | 21.7650 | 52.7020 | 75.2603 |
| Delta (percentage points) | +101 | +0.0992 | +0.0496 | +0.0000 |

Both arms: seed42, six full epochs, 6086 presentations/epoch, 381 batches/epoch, 2286 updates; same frozen DQU, v7 and unchanged 2017-query/3817-image validation. Best epoch5 selected by R@10+R@50. Original arm pads with 101 original repeats; expanded arm uses all 6086 unique records.

The change equals net +2 queries at R@1, +1 at R@10, and 0 at R@50. This is a small single-seed pilot, not evidence of stable or statistically significant improvement. Shirt/Toptee were not retrained. No extra training or generation launched.

101 new structured records completed online: 98 valid, 3 DQU fallback. 109 metered requests, estimated list-price cost CNY0.0694666; not an invoice.

Audit: both exit codes0; six completed epochs each; finite saved tensors; checkpoint metrics match JSON; frozen DQU, source/code hashes, shared original structures and heldout annotations unchanged. GPU released, useful weights retained.
