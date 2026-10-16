# CIRR / Shoes：DQU 基线与当前方法

## 已确认的范围

新分支 `codex/cirr-shoes-benchmarks` 从噪声审计分支继续，保留 FashionIQ 默认行为。
比较顺序是：每个数据集单独训练原 DQU → 冻结该数据集 DQU，训练结构化文本残差 → identity / annotated 多正样本消融。
所有分支使用同一完整评估查询和图库。缺少或未通过检查的结构化结果退回 DQU；报告结构化覆盖率，不筛掉评估查询。

2026-10-15：Shoes 图片已准备好并启动原 DQU 训练；CIRR 注释已准备好，但原始图片仍缺失。
本文的协议核验和单元测试不能当作最终检索结果。尚不能声称方法在新数据集上有效。

## 官方版本及数据约定

- [DQU 固定版本](https://github.com/iLearn-Lab/SIGIR24-DQU-CIR/tree/49f53992447a9a84b897d9b2e2f2c91dd2cc8788)：`49f53992447a9a84b897d9b2e2f2c91dd2cc8788`。
- [CIRR 注释固定版本](https://github.com/Cuberick-Orion/CIRR/tree/2276fba2d2c53862c8ae79ea18e1f1ae03719f56)：`2276fba2d2c53862c8ae79ea18e1f1ae03719f56`。
- [DQU README](https://github.com/iLearn-Lab/SIGIR24-DQU-CIR#datasets) 包含所需 BLIP2 描述、关键词及 Shoes 图片包链接。
- [CIRR 图片申请说明](https://cirr.zheyuanliu.me/raw-image-download)：需先申请 NLVR2 原始图片访问；图片未到位时不启动训练。

| 数据集 | 训练查询 | 评估查询 | 训练图片 | 评估图库 | 原 DQU 学习率 / CLIP 学习率 |
|---|---:|---:|---:|---:|---|
| Shoes | 8,990 | 1,761 | 10,000 | 4,658 | 5e-5 / 5e-6 |
| CIRR | 28,225 | 4,181 | 16,939 | 2,297 | 1e-4 / 1e-6 |

共同参数：ViT-H/14 LAION 权重、batch 16、seed 42、原 DQU 融合与单向 NCE，最多 100 epoch，5 次不提升后停止。
服务器是 RTX 4080 SUPER（实际报告 32,760 MiB）、Torch 2.9 / OpenCLIP 2.23；与论文环境不完全相同，属于同环境受控对比，单 seed 不能支持统计显著性结论。

### Shoes

使用 `image_captions_shoes.json`（BLIP2），不能误用 `captions_shoes.json`。
按 `train_im_names.txt` / `eval_im_names.txt` 切分，读取 `womens_*/*/*.jpg`，执行官方拼写修正，关键词键为 `reference+target`。
参考图先缩放至 512×512 再绘字；文字严格为 `BLIP2 caption + ', but ' + modification`。
评估去除参考图，报告 R@1/10/50，以 R@10 + R@50 选模型。

官方数据有 26 条训练、1 条评估的 reference==target。保留它们以匹配原 DQU；那一条评估的目标也会按官方规则被去除，因此不可能命中。
允许它们作为自身查询的已标注正样本，但禁止把参考图传播为其他查询的额外正样本。
共有 6,633 个不同训练目标 ID，4,616 条训练查询的目标被其他查询重复使用，故 identity 消融有实际意义。

### CIRR

同时支持官方平铺注释布局和原 DQU 的 `captions/captions`、`captions/image_splits` 布局；图片支持 `images/`、`img_raw/` 或根目录下的 train/dev 结构。
禁止跨 split、路径越界、含糊的重复图片布局和无效六图子集。
保留原 DQU 的 `', but'` 拼接（无额外空格），训练和验证图片路径统一解析，绘字前不强制缩放 512。
使用全库 R@1/5/10/50 和子集 R@1/2/3；子集从去参考图后的全库排序中筛出，避免另行排序造成同分顺序变化。
选模匹配原 DQU 代码的七项召回率之和，额外报告常用 `(全库 R@5 + 子集 R@1)/2`。
不载入 test1 图片或目标标签，不用测试集选模型；本分支暂不实现在线 test1 提交。

## 如何避免沿用错误的 FashionIQ 假设

新 `BenchmarkData` / `CIRBenchmark` 由原基线和改进分支共享，旧 `datasets.py` 和 `newDataset.py` 中的 CIRR/Shoes 类不作为新训练入口。
验证图片按批读取，排序阶段直接使用元数据，避免旧实现把所有查询/图库预先存成图像张量。
数据集权重必须来自相同数据集；Dress 权重不能标作 Shoes/CIRR 基线。

`generate_benchmark_edits.py` 只向 Qwen 提交参考图片和修改文本，目标 ID 只作为结果绑定元数据。
它保留 CIRR 的人物、场景、动物、数量和位置，也保留 Shoes 的鞋跟、鞋面等信息；不使用 FashionIQ 的服装排除规则、噪声策略或语义清洗器。
结构化 JSON 以原查询索引、两图 ID、修改文本、类别和 split 共同绑定，不能仅凭图对合并不同指令。
结果需通过 schema 检查及单独的 query-only Qwen 语义自检；自检是模型信号，不是人工真值。
生成失败会保留失败记录，训练回退 DQU，不伪造有效标签。

2026-10-16 另完成 [在线 Qwen3.7-Flash 的小规模文本对照](shoes_online_structured_pilot.md)：24 条提示词/自检对照及 8 条事后回归检查，共 112 次请求，标价估算 0.0377424 元。其结果与代码独立保存，未替换本轮本地 Qwen3-VL 生成，也未增加检索训练；格式通过和自检通过不代表真实检索提升。

**后续更新：** 用户明确要求停止本地 Shoes 生成，现已切换到 [全量在线 Qwen3.7-Flash 队列](shoes_online_full_20261016.md)。旧本地 1,452 条记录保留；正式训练路径由 `experiment_report.json.structured_paths` 指向 `online_qwen37_v3/`，不能读取根目录旧文件冒充在线进度。完整 train/val 后自动运行原三组 6 轮比较，基线不重训。

训练集精确同参考图、同修改文本的额外标注正样本覆盖：Shoes 64/8,990；CIRR 170/28,225。
这是保守的已有标注扩展，不把 CLIP 相似图片或 CIRR 子集所有图片直接视为正样本。
另检查了 CIRR 官方 `target_soft`：其中同时存在 1.0、0.5、0.2 和 -1.0，且有条目包含参考图。
不能把这个字典的所有键直接当作额外正样本。本轮 identity/annotated 不使用 `target_soft`；后续需按其标注定义单独设计软标签消融。

## 运行

首先用纯元数据审计，无需 GPU：

```bash
python src/benchmark_data.py --dataset shoes --root /root/autodl-tmp/data/Shoes
python src/benchmark_data.py --dataset cirr --root /root/autodl-tmp/data/CIRR
```

完整顺序实验由 `scripts/run_cirr_shoes_benchmark.py` 管理：原 DQU → Qwen train/val → 结构化 off/identity/annotated。每阶段单独进程，避免大模型同时驻留显存。

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 \
python scripts/run_cirr_shoes_benchmark.py \
  --dataset shoes --data-root /root/autodl-tmp/data/Shoes \
  --output /root/autodl-tmp/results/cirr_shoes_20261015 \
  --clip-checkpoint /absolute/path/open_clip_pytorch_model.bin \
  --qwen-model /absolute/path/Qwen3-VL-snapshot
```

先加 `--dry-run` 检查计划。已有**完成的**同数据集原 DQU checkpoint 时用 `--skip-baseline`。
默认生成全量结构化数据；`--generation-limit N` 明确标为低覆盖率 pilot，不能冒充全量实验。
adapter 默认 6 epoch、冻结基线；hard-negative 权重设 0，以单独观察多正样本效果。
完整比较应再增加匹配预算的 DQU 继续训练对照及多个 seed，排除额外训练预算和随机性。

默认要求 checkpoint 文件系统至少 8 GiB 可用，每个 epoch 前检查；不足即抛出 STOP，停止后续阶段。
只保存最佳权重，不保存 AdamW 状态；先写临时文件再原子替换，保存失败保留此前最佳模型。
adapter checkpoint 依赖对应原 DQU 大权重，不能删除其 base checkpoint。

## 本次服务器清理及验证

初始数据盘空闲约 9.4 GiB。发现 Qwen3-VL 四个分片同时存在实体 snapshot 和 blobs 副本。
逐一计算两份 SHA-256，并确认与 blob 文件名一致后，将 snapshot 改为指向相同 blob 的硬链接。
释放 **17,534,339,512 bytes（16.33 GiB）**，所有模型路径仍可读取；没有删除实验 checkpoint。
准备 Shoes 图片和注释后约剩 25.3 GiB，足以顺序运行 Shoes。

Shoes 图片包 SHA-256：`da66f14692c94b48dd4d8f782d7431fab84dcd929afbb3af76030f8a149564bf`。
包内 14,764 张 JPG 全部解码检查通过；正式切分使用其中 14,658 张，其余不加入图库。
官方 dataset 源码 SHA-256：`dd087591d505f6cfde25275d35a2dfb3eb34eb9a8b9c988d6ea291adfc31ceed`。
训练/评估全部 10,751 条查询的参考 ID、目标 ID、归一化文本与官方一致，抽查 6 条绘字图及目标图像素完全相同。
上述核验不等同于论文数值复现，最终指标以完成训练后的 checkpoint JSON 为准。

运行目录：`/root/autodl-tmp/experiments/cirr-shoes-20261015`；结果目录：`/root/autodl-tmp/results/cirr_shoes_20261015`。
基线日志：`shoes_dqu_original.log`。新代码的最终快照和报告会另外记录，避免修改正在训练的源文件。

当前已启动后续队列，使用最终快照 `/root/autodl-tmp/experiments/cirr-shoes-final-20261015`。
队列等待基线进程结束，并检查正常完成日志后再使用 GPU；失败则不继续。
队列日志为 `queue.log`，实时状态为 `experiment_report.json`；已完成的中间评估快照记录在本仓库 `results/cirr_shoes_20261015/progress_snapshot.json`，它不是最终结果。
本地和服务器均有 37 项测试通过；另用明确标为假模型的 CPU 集成测试遍历完整 Shoes 查询和图库，核验两个评估入口在同特征下的召回率一致。
