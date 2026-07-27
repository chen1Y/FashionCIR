# DQU-CIR baseline

The `dqu*` files are an isolated reproduction of the FashionIQ path in the
official [SIGIR24-DQU-CIR](https://github.com/iLearn-Lab/SIGIR24-DQU-CIR)
repository. They do not import the locally modified `datasets.py` or the
structured-edit `new*` implementation.

The reproduction preserves the baseline's method:

- ViT-H/14 (`laion2B-s32B-b79K`);
- BLIP-2 reference caption plus the corrected original modification as the
  unified textual query;
- the reference image with the extracted target keyword written onto it as the
  unified visual query;
- adaptive scalar fusion with dropout 0.5;
- CLIP learning rate `1e-6`, fusion learning rate `1e-4`;
- batch size 16 and one-way batch NCE initialized with a logit multiplier of 10.

Only two compatibility changes are intentional:

1. both flat `resized_image/<id>.jpg` and the original nested
   `resized_image/<category>/<id>.jpg` layouts are accepted;
2. `--clip-checkpoint` can point to the already downloaded OpenCLIP weight file.

Run Dress on the paper's full validation gallery:

```bash
cd src
python dquTrain.py \
  --dataset dress \
  --fashioniq-split original-split \
  --fashioniq-path ../data/FashionIQ \
  --clip-checkpoint /path/to/open_clip_pytorch_model.bin \
  --batch-size 16 \
  --seed 42 \
  --run-name dqu_official_original
```

The best epoch is selected by `R@10 + R@50`. Its metrics JSON and checkpoint
are written under `--model-dir`. The checkpoint intentionally omits AdamW
optimizer states to avoid consuming several additional gigabytes.

For the smaller paper-specific validation gallery, change the split to
`val-split`. Paper values are five-seed means, so a strict reproduction should
run both protocols with five seeds and report mean and standard deviation.
