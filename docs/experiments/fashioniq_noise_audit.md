# FashionIQ 噪声审计与训练隔离

## 处理思路

FashionIQ 原仓库的 [issue #6](https://github.com/XiaoxiaoGuo/fashion-iq/issues/6)
已有非服装 LOGO 图片与两条修改文本不一致的报告。我们此前的等价查询审计也发现了
“不是服装”“商品图是文字”等描述。先识别噪声来源，再决定排除、保留或修复。

| 问题 | 初版检测 | 处理 |
|---|---|---|
| 文件缺失、JPEG 损坏、非法 ID、错误标注结构或 split 归属 | 文件解码、结构和成员检查 | 可生成自动训练排除项 |
| 纯 LOGO、文字页、尺码表、其他商品 | 已有 OpenCLIP 图文相似度初筛 | 复核后排除该图关联的全部训练查询 |
| 衣服上有 LOGO/文字 | 单独设置 printed-garment 正类提示 | 保留为服装候选，检查误报 |
| 模糊、尺寸极小、低信息图片 | 尺寸、灰度方差、熵 | 仅提示复核；纯色衣服可能误报 |
| 同图不同 ID、重复标注 | 文件 SHA256、三元组内容哈希 | 仅报告；多视图或同商品不能直接算错 |
| 修改文本矛盾、目标不满足修改、参考图异常 | 文本启发式和双图 Qwen 审核 | 输出逐句证据及不确定状态，复核后隔离查询 |

CLIP 的自然语言分类来自 [CLIP 论文](https://arxiv.org/abs/2103.00020) 中的零样本分类思路。
本实现比较若干服装类别（含印花衣物）与独立 LOGO、文字页等负类的原始余弦分数。
默认分差阈值 0.03 是可调整的筛查启发式，**不是经过校准的噪声概率或可信删除阈值**。
需在 train 的人工复核子集上检查精确率和召回率，并包含正常、印花 LOGO、纯色服装等对照。

本次代码默认保留原数据，只输出审计报告和排除清单。原始文件不删除、不搬移、不覆盖。
清单只过滤 train，验证查询和图库保持官方原样。单独审计 val 可用于诊断，不能编译成训练清单，
也不能通过删掉难例或 LOGO 验证图来报告更高的“官方”Recall。

## 代码入口

- `src/fashioniq_noise.py`：图像/文本审计、审核决策、清单编译和有效性检查。
- `src/audit_fashioniq_noise.py`：批量扫描、可选 CLIP 初筛、图片和三元组预览页、审核模板。
- `src/review_fashioniq_triplets.py`：本地 Qwen3-VL 双图语义审核，支持断点继续，输出模型建议。
- `newTrain.py` 和 `dquTrain.py`：新增可选 `--noise-manifest`。默认关闭，原模型和评估协议不变。

这里允许离线审核模型看到训练集 reference、target 和两条原始 caption，因为目标是核对已有监督。
它与 `generate_structured_edits.py` 的查询解析任务分开；查询解析仍只使用参考图和修改文本。
审核结果不会作为检索输入，也不会自动作为正确标签或排除决策。

## 1. 全量审计

从仓库根目录运行，输出目录须尚不存在：

```bash
python src/audit_fashioniq_noise.py audit \
  --fashioniq-root /root/autodl-tmp/data/FashionIQ \
  --output-dir /root/autodl-tmp/results/noise_audit_new \
  --clip-checkpoint /path/to/open_clip_pytorch_model.bin
```

默认扫描 dress/shirt/toptee 的 train 图像划分，包括没有出现在三元组中的图。
支持平铺和 `resized_image/<category>/` 布局。省略 `--clip-checkpoint` 可只做 CPU 文件/文本检查。
`--limit-images 100` 用于检查流程，但部分扫描结果不能编译训练清单。
`--split val` 仅生成诊断报告；不会写入数据集目录。

每个类别输出：

- `audit.json`：逐图证据、疑似三元组、重复组和统计。
- `image_review_*.jpg`：疑似图片预览，标明图片 ID、CLIP 类别与分差。
- `triplet_review_*.jpg`：参考图、目标图、两条修改文本及触发规则。
- `review_template.json`：所有初始决定均为 `pending`。
- `semantic_review_requests.jsonl`：用于进一步审核的图像对任务。

图像候选、三元组候选以及规则命中数的分母不同，不应混为“噪声比例”。

## 2. 双图语义审核

先做少量可行性检查。默认只使用本地缓存，拒绝 val 任务，不调用云 API：

```bash
python src/review_fashioniq_triplets.py \
  --requests /root/autodl-tmp/results/noise_audit_new/dress/semantic_review_requests.jsonl \
  --output /root/autodl-tmp/results/noise_audit_new/dress/qwen_review_proposals.json \
  --model Qwen/Qwen3-VL-8B-Instruct --limit 12
```

模型逐句返回 `satisfied / violated / unknown`、两图是否为服装、caption 是否冲突、
`keep / exclude / uncertain` 建议及证据。不确定、遮挡、稀疏描述和多种合理答案不直接判为错误。
输出种类为 `model_review_proposals`，与训练排除清单的格式不同，不能直接交给训练器。
模型权重或图片不存在、响应格式错误、OOM 都记录/报告为失败，不会被当成干净或噪声标签。

人工或独立审核者应对照图片检查模型建议。不能只用训练模型自己的高损失、低相似度或自信度
判定一个三元组错误，否则容易把困难但正确的样本清掉。

## 3. 审核与编译排除清单

复制 `review_template.json` 为自己的审核文件。对复核完成的项填写：

```json
{
  "decision": "exclude",
  "reviewer": "reviewer_name_or_explicit_model_identity",
  "reason": "A standalone brand wordmark is visible; no garment is present."
}
```

保留模板原有图片 ID、文件 SHA256、查询原始行号、行内容 SHA256 和数据指纹。
允许 `pending / keep / exclude / uncertain`；pending 和 uncertain 不产生语义排除。
最终 keep/exclude 决定需要审核者身份和可核查理由。记录模型审核时要明确标明模型身份，
不要把模型判断声称为人工真值。

图片级 exclude 表示该图本身不适用，编译器排除所有以它作为参考或目标的训练查询。
查询级 exclude 只隔离指定三元组，其图片可继续用于其他正常查询。

```bash
python src/audit_fashioniq_noise.py compile \
  --fashioniq-root /root/autodl-tmp/data/FashionIQ \
  --audit /root/autodl-tmp/results/noise_audit_new/dress/audit.json \
  --reviews /root/autodl-tmp/results/noise_audit_new/dress/reviewed.json \
  --output /root/autodl-tmp/results/noise_audit_new/dress/train_filter.json
```

省略 `--reviews` 仅排除明确的文件/结构故障。CLIP、文本关键词、低信息和重复提示都不会自动升级为删除。
默认排除训练查询超过 20% 时中止，供检查目录是否错误或审核是否过宽；确有需要可显式修改
`--max-exclusion-fraction`，编译器始终禁止清空整个训练集。

清单绑定 cap JSON、split JSON、纠错字典的文件指纹，所有排除行的内容哈希，以及被排除图的文件哈希。
标注重新排序、划分变更、被排除图已恢复/修改时会拒绝旧清单，要求重新审计。

## 4. 训练对照与多正样本兼容

在原命令后添加清单路径：

```text
--noise-manifest /root/autodl-tmp/results/noise_audit_new/dress/train_filter.json
```

先用同 seed、同模型、同预算比较原训练与过滤训练。报告排除数量、原因、剩余查询数和官方全量验证指标。
不要同时改变文本重写、模型结构或多正标签来源，否则无法判断改善来自哪一步。

过滤在读取 train 行时执行，保留原 source_index，不给幸存行重新编号。
多正集合在过滤后重建；经审核的正例文件若引用了已排除的查询或图片，训练会报错，要求同步修订。
这能避免前一轮多正实验把问题目标重新加入候选库。

修复 caption 或替换错误 target 需要另建带原值、审核证据的标注补丁，并重生成相关 Qwen 结构化记录、
BLIP 描述和关键词缓存。本初版提供隔离训练路径，没有自动改写或猜测正确 target。

## 验证

本地与服务器均已通过 28 项测试，包含原有 17 项多正测试和新增 11 项噪声测试。
覆盖过滤前后索引稳定、验证不受过滤影响、排除图的全部关联查询、修复文件后的清单失效、
误报保护、审核身份与证据、原始标注结构异常及两种图片目录布局。

```bash
python -m unittest discover -s tests -v
```

实际数据结果见 [真实数据审计记录](fashioniq_noise_results.md)。本次未进行清洗后的模型训练，
未声称检索收益；服务器 Qwen 离线缓存不完整，双图语义模型试跑未产出判断。
