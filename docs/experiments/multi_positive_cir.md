# FashionCIR 多正样本实验（初版）

## 思路与论文依据

组合查询的正确答案可能不止一张。比如“保留参考裙子的款式，改成蓝色”，
图库中可能有多个合理目标；仅因它们不是原始 target，就把它们推远，会造成假负样本。
不过“都是蓝色裙子”也不足以成为正样本：还要满足应保留的款式和全部修改条件。

- [SEARLE / CIRCO](https://arxiv.org/abs/2303.15247) 提供一个查询对应多个正确目标的评估设定；
  [官方数据集](https://github.com/miccunifi/CIRCO) 明确以缓解现有数据集的假负样本问题为动机。
- [SPN](https://arxiv.org/abs/2404.11317) 用多模态大模型构造更多 CIR 三元组，扩充正例训练数据。
  这与“同一查询有多个目标”相关，但不是同一件事；本实现不是 SPN 复现。
- 损失借鉴 [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362)：
  对每个已确认正例的 log probability 求平均，而不是只要求一个最容易的正例得分高。

**本项目适配方案**：先修复同目标 ID 的假负例，再利用已有精确等价查询的标注，
最后接入经审核的额外真实目标。现有 Qwen 的 add/remove/relations/target description
适合用于候选筛选和审核，但不能单凭相似度或 Qwen 自信度把候选直接当成真实正例。

## 已实现的范围

接入 `newTrain.py` / `newDataset.py` / `newModel.py`，不修改隔离的 `dqu*` 基线和评估代码。

| 参数 | 正例来源 | 用途 |
|---|---|---|
| `--multi-positive off` | 原来的单 target | 默认；回归对照 |
| `--multi-positive identity` | batch 中 target ID 相同的所有列 | 消除相同图像 ID 的错误负例 |
| `--multi-positive annotated` | identity + 相同参考图、相同纠错后修改文本的已有目标 | 无需新增伪标签的保守实验 |
| `--multi-positive verified --positive-manifest PATH` | annotated + 审核文件 | 对同一查询补充真实正例 |

精确匹配不做文本语义聚类，不跨参考图合并，不从“相同目标的另一条查询”传播额外标签。
新增标签绑定原始 train 行号、candidate、target 和纠错后文本，防止文件顺序改变后错配。
所有训练查询及新增目标必须出现在该类别的 `split.*.train.json`，且正例不能是参考图本身。

默认每条多正查询随机抽取 **一张额外正例**。它保留原参考图、原修改文本和原 Qwen 输入，
只扩充候选目标图片张量，因此原有依赖 candidate/target 的写字图不会被替换成错误输入。
不足的槽位被剔除后再编码。B 条查询的候选库最多为 2B 张图，计算和显存开销会增加。
也可用 `--no-sample-extra-positive` 仅在原 batch 内应用多正标签。

设查询 i 的正例列集合为 P_i，logit 为 z_ij：

```text
L_i = - mean_{p in P_i} log softmax(z_i)[p]
```

正例只有对角线时退化为原始单向交叉熵。重复目标的不同图像增强各作为一列参与计算，
正例列等权；暂未做按唯一图像 ID 重加权，比较实验时需要记录这种差异。

三个训练信号同步调整：

1. 排序 NCE：对所有已确认正例求平均。
2. 门控教师：比较原文本和结构化文本时使用相同的正例集合。
3. 难负样本：排除所有正例，以平均正例相似度计算 margin；不足 k 个负例时只用实际存在的负例。
   没有负例的行安全跳过。难负损失仍只通过现有残差分支更新。

运行日志输出多正查询实际覆盖率（按不同目标 ID 统计）和平均候选库大小；
checkpoint 的 config 保存实验开关。推理模型结构和 checkpoint 参数名不变。

## 使用方法

以下从 `src` 目录运行，沿用已有 CUDA / OpenCLIP 训练环境。

先审计已有标注，无需模型、图片或 PyTorch：

```bash
python multi_positive.py --fashioniq-root ../data/FashionIQ --category dress --mode annotated
```

如需新目标审核文件：

```bash
python multi_positive.py --fashioniq-root ../data/FashionIQ --category dress \
  --mode annotated --review-template ../data/FashionIQ/dress_positive_review.json
```

模板只复制已有查询，`positive_targets` 为空且 `reviewed=false`。
它不会寻找或虚构新正例；需要在 train 图库选候选并审核后填写。文件已存在时拒绝覆盖。
仅保留审核完成的行，再将这些行标为 true，其余行删除。

文件结构示意（ID 和描述均为虚构示例，不可直接用于训练）：

```json
{
  "schema_version": 1,
  "category": "dress",
  "split": "train",
  "queries": [
    {
      "source_index": 12,
      "candidate": "reference_id",
      "target": "original_target_id",
      "modification_text": "make it blue and keep the same shape",
      "reviewed": true,
      "positive_targets": ["reviewed_blue_dress_id"]
    }
  ]
}
```

新增目标必须同时满足：服装主体及应保留特征合理、add 全部满足、remove 对应的旧特征消失、
相对关系正确。比较长度/颜色深浅等条件时必须看参考图；仅凭目标描述相似不可通过。
不确定的候选不标正，也不因未审核就声称是真负例。初版仍将未标注候选按原流程处理，
因此尚未彻底解决所有假负例问题。

从已有、已验证的 DQU checkpoint 开始，先跑两批 smoke test：

```bash
python newTrain.py --dataset dress --fashioniq-split original-split \
  --fashioniq-path ../data/FashionIQ \
  --clip-checkpoint /path/to/open_clip_pytorch_model.bin \
  --dqu-checkpoint /path/to/dress_dqu_official_original_seed42_best.pt \
  --multi-positive annotated --run-name mp_annotated_smoke \
  --max-train-batches 2 --num-epochs 1 --num-workers 0 --batch-size 4
```

正式训练去掉批次限制，使用与现有对照完全相同的 batch size、Qwen JSON、written-keyword-source、
数据范围和训练预算。审核版将实验开关改为：

```text
--multi-positive verified --positive-manifest ../data/FashionIQ/dress_positive_review.json
```

无需在 `newTest.py` 增加参数。标准 FashionIQ 验证仍只使用官方 target、完整 original-split
图库和原有去参考图规则。训练中的新增正例不能修改验证真值。

## 建议实验顺序

| 对照 | 唯一新增因素 | 需要关注 |
|---|---|---|
| A | off | 当前结构化分支的相同训练配置 |
| B | identity | 同 target ID 冲突究竟有多少 |
| C | annotated，关闭额外采样 | 精确等价查询能提供多少可见正例 |
| D | annotated，打开额外采样 | 显式增加正例是否比偶遇有效 |
| E | verified，打开额外采样 | 新增审核标签的价值 |

若 annotated 审计显示多正查询为 0，就跳过 C/D；不要通过放宽到“文本相似”制造正例。
在 train 中预留开发子集决定候选阈值和审核策略，不在 dress 验证集反复调参。
至少使用 42/43/44 三个种子；报告 R@1/10/50、均值/标准差、正例覆盖率、训练时间和显存。
这是多正监督消融，并不自动代表新的论文级方法创新。

如果要研究多答案评估，另建人工审核的 FashionIQ 子集，完整复核各模型的候选池并报告
mAP@K / 多正 recall；它属于扩展评估，不能替代官方单 target 指标，
也不能仅给本方法检索结果补标、让对照方法处于不利条件。

## 本地验证与局限

本地仓库没有 FashionIQ 数据、CLIP 权重或训练 checkpoint。
初版先验证实现、合成张量上的损失/梯度和数据路径；后续服务器试验见下文。
测试无需下载 CLIP，但数值测试需要 PyTorch：

```bash
python -m unittest discover -s tests -v
```

2026-10-14 本地验证：Python 3.12 / PyTorch 2.14.1+cpu，17 项测试全部通过；
修改文件语法编译及 `git diff --check` 通过。该本地验证不包含真实 FashionIQ GPU 训练或验证集评估。

覆盖单正退化、额外正例梯度、矩形候选库、难负例排除、无负例/无有效行、
半精度大 logit、完整 compute_loss 反向传播、额外图片与 padding、审核文件隔离和陈旧标签拒绝。

## Git 分支

实验在 `codex/multi-positive-cir`，`main` 保留原代码。
分支是同一仓库中的独立修改路线，便于比较和撤回；无需复制整个 GitHub 仓库。
此处仅提交代码、测试与说明，不提交数据、模型权重或真实审核标签。

## 服务器试验入口

已完成的两组 Dress 初步实验见 [服务器结果报告](multi_positive_server_results.md)。
当前多正版本未改善主指标，保留为实验分支。

`scripts/run_multi_positive_pilot.py` 按顺序运行等预算对照，并保存命令、代码 SHA256、
日志、耗时和最佳指标。默认比较 `off` / `annotated`；`identity` 可显式指定。
输出目录已有 `pilot_status.json` 时拒绝覆盖。

```bash
python scripts/run_multi_positive_pilot.py \
  --fashioniq-path /root/autodl-tmp/data/FashionIQ \
  --clip-checkpoint /path/to/open_clip_pytorch_model.bin \
  --dqu-checkpoint /root/autodl-tmp/checkpoints/dress_dqu_official_original_seed42_best.pt \
  --output-dir /root/autodl-tmp/results/multi_positive_new_run \
  --modes off annotated --seeds 42 --num-epochs 6
```

试验固定 `original-split`、batch size 16、两轮 gate 预热和四轮 adapter 训练；
hard-negative 权重设为 0，与 v7 设定对齐，优先隔离多正 NCE 和门控教师的影响。
这组试验不能用于评价非零 hard-negative 权重的增益。

2026-10-14 服务器原始 train 标注审计：

| 类别 | 查询数 | 有不同正例目标的查询 | 覆盖率 | 总正例关系数 |
|---|---:|---:|---:|---:|
| dress | 5985 | 55 | 0.9190% | 6049 |
| shirt | 5988 | 96 | 1.6032% | 6096 |
| toptee | 6027 | 38 | 0.6305% | 6079 |

三个类别都满足“不同训练行的 target ID 不重复”。因此 `identity` 在当前数据上没有
可修复的同 ID 冲突，不应声称它能提升此数据集。Dress 的 55 条查询来自 26 个精确等价组；
它们都有通过现有解析规则的结构化记录，但这不等于标注已经过人工语义复核。
例如部分重复查询描述“不是服装”或“商品图片是文字”，需要单独审查数据噪声。

新增正例来自多个三元组的已有目标并集，而不是给任意单个三元组凭空生成标签。
没有精确等价三元组的查询仍然只有一个标注目标。若要覆盖更多查询，需另外进行候选挖掘、
语义复核及 `verified` 文件标注；本次没有开展这项新增标注。

本机 GitHub 认证和推送步骤见 [Windows 推送说明](../github_push_windows.md)。
