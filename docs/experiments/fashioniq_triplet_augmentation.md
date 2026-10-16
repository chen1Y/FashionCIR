# FashionIQ：使用现有训练图片扩充三元组

2026-10-15，分支 `codex/fashioniq-triplet-augmentation`。

## 已完成的小规模尝试

| 项目 | 原始 Dress | 本次覆盖层 |
|---|---:|---:|
| 训练图片 ID | 11,452 | 11,452 |
| 训练三元组 | 5,985 | 6,005 |
| 验证查询 | 2,017 | 2,017 |
| 验证图库 | 3,817 | 3,817 |
| 新增物理数据图片 | — | 0 |

CPU 生成 24 个候选新图对，Codex 查看双图预览后保留 20 对，拒绝 4 对衣物类别或图像范围不清的候选。
这些是**单模型视觉检查后的合成伪标注**，没有独立人工标注或独立模型审校，20/24 不是精度估计。
原始标注、图片和所有类别的验证/测试元数据均保留；新增样例不进入验证/测试集。

服务器输出：`/root/autodl-tmp/results/fashioniq_augmentation_20261015/overlay`。
独立代码目录：`/root/autodl-tmp/experiments/fashioniq-augmentation-20261015`。
结果见仓库 `results/fashioniq_augmentation_20261015/pilot_report.json` 和 `reviewed_additions_dress.json`。
服务器 47 项测试全部通过；`dquDataset.FashionIQ`、`newDataset.FashionIQ` 均实际读取了 6,005 条训练数据，
20 条新增样本均成功解码、绘制关键词，原始训练数据前缀保持一致。
本地 Windows 无符号链接权限，该项测试跳过；Linux 服务器已完整验证。

**尚未训练检索模型或测量召回率提升。** Shoes 队列占用 GPU，本次仅进行 CPU 筛选、视觉检查和数据加载验证。
可选的 Qwen 双图生成器尚未进行真实模型推理验证；其占用保护已有单元测试。

## 方法：固定图片池，增加查询边

三元组是 `(参考图片, 修改文本, 目标图片)`。固定图片节点后，可以增加未出现过的有向图对及对应文本。
例如，本次新增 `B00CHUH8MY → B0009Y2LN6`：

```json
{
  "candidate": "B00CHUH8MY",
  "target": "B0009Y2LN6",
  "captions": [
    "change the black dress to purple and shorten it to below the knee",
    "add short lace sleeves and a V neckline"
  ]
}
```

两张照片原本就位于 Dress 训练图片池内。参考图是黑色无袖长裙，目标图是紫色、有短蕾丝袖和 V 领的较短连衣裙。
没有生成目标图片，也没有将同一图对简单改写几句文本来计数。
新三元组不等于多正样本：一个查询是否还存在其他有效目标，需要另行验证，不能自动把相似图片全部当正样本。

## 数据与质量约束

1. 在同一类别的官方训练图片中取图，并排除所有类别 val/test 图片 ID 的并集。缺少任一 held-out split 文件即停止。
2. 用已有 BLIP 图片描述的词汇重合度提名同类、颜色不同的图对；这只是文本筛选，不是 CLIP 视觉近邻，也不能证明相对描述正确。
3. 可读取绑定原数据指纹的噪声审计，将有标记的图片排除出本次候选池；不据此删除原图或原三元组。本次排除 761 个带标记的采样候选。
4. 禁止自身配对、已有有向训练图对、重复新增图对、跨 split 图对及字节完全相同的照片。
5. 双图检查衣物类别、修改方向、每句描述是否对目标成立、是否至少存在一个明确变化；不根据人物、姿势、背景或不可见材质猜标签。
6. 每个参考或目标最多新增 2 条边，默认新增量不超过原训练条数的 10%。不确定样例保持 pending 或 reject。
7. 清单绑定原始 JSON 元数据及两张图片 SHA256。导出时重新检查；源数据变化后必须重新生成/检查。

这些结构校验不能代替语义审校。BLIP 可能漏掉款式细节，单模型也可能自信地写错；正式规模扩充应抽样人工检查，
并优先选择视觉相近、差异少、修改可以完整描述的图对。当前 20 条覆盖多种差异，只是数据管线试验。
后续可以加入现有 CLIP 的同类近邻、差异数量分层和独立视觉模型复核；这些尚未实现，不能当成本次实验结果。

## 接入已有 DQU 和结构化分支

`src/fashioniq_augmentation.py` 创建一个独立元数据覆盖层：

- `resized_image` 和 `image_splits` 是原目录的符号链接，没有复制图片。
- 训练标注保持原 5,985 条为前缀，再追加新样例；其他类别训练标注及验证/测试标注链接回原文件。
- `keywords_in_mods_dress.json` 保留原键，新图对增加 `candidate_target` 键。
  其最后一项只含被描述支持的目标正属性，用于原 DQU 在参考图上绘字；不会把被删除的源属性画进去。
- 原始数据加载器无需修改，只需将 `--fashioniq-path` 指向覆盖层。

**现有结构化 JSON 不包含这些新图对。** `newDataset` 会让它们回退到 DQU 输入；`structured_only=True` 会筛掉新图对。
如果冻结 DQU 且只训练结构化残差，这些缺少结构化输入的样例不能直接训练该残差。
首先适合在 DQU 可训练的配置中比较原始数据与原始+新增数据；要训练结构化分支，需要再用
“参考图+新增修改文本”生成结构化编辑，不能把目标图传入这个结构化生成步骤。
本次双图只用于构造/检查训练标签，这是离线训练数据合成，不能用于验证查询或测试推理。

旧噪声过滤清单绑定原训练 JSON 指纹，不能直接复用于覆盖层；需要针对扩充后标注重新审计并编译。

## 复现候选与覆盖层

在独立代码目录中执行；输出使用新的目录/文件名，避免覆盖本次证据。

```bash
python src/fashioniq_augmentation.py propose \
  --root /root/autodl-tmp/data/FashionIQ --category dress --seed 42 --limit 24 \
  --noise-audit /root/autodl-tmp/results/noise_audit_20261014/dress/audit.json \
  --output /root/autodl-tmp/results/augmentation_retry/proposals.json

python src/fashioniq_augmentation.py sheet \
  --root /root/autodl-tmp/data/FashionIQ \
  --manifest /root/autodl-tmp/results/augmentation_retry/proposals.json \
  --start 0 --output /root/autodl-tmp/results/augmentation_retry/pairs_0.png

python src/fashioniq_augmentation.py build \
  --root /root/autodl-tmp/data/FashionIQ \
  --manifest results/fashioniq_augmentation_20261015/reviewed_additions_dress.json \
  --output /root/autodl-tmp/results/augmentation_retry/overlay
```

清单中每条接受样本包含两句 `captions`、`target_keywords`、两张图片哈希以及明确的 `review`。
审校者和证据必须如实记录，未知不能填为 true。
本次原始元数据签名：`001866003f5b621c6be4d222826c6164e98f609ee8a3b739265873e593c89a41`。
若源元数据发生变化，不能手工替换签名绕过检查。

## 可选：用现有 Qwen 自动生成并复核

**2026-10-16 更新：** 用户已授权在 Shoes 改用在线生成后，利用空闲 GPU 提前处理原定 120 对。已实施 [有身份验证和自动恢复的 GPU 窗口协调](fashioniq_online_window_20261016.md)，不再要求这一批等 Shoes 全部结束；默认严格守卫仍保留。旧“只在队列结束后”的示例适用于未提供协调证明的直接运行。

`src/generate_fashioniq_pair_captions.py` 对每个待审图对执行两个独立提示词调用：
先生成两句相对描述和目标关键词，再让同一权重重新查看双图、检查每项声明。
第二次调用不携带生成理由，但同模型的两次调用不构成独立验证。保留原始响应；不通过的样例拒绝导出。

仅在现有 benchmark 队列结束且 GPU 无计算进程后手动执行，例如：

```bash
python src/generate_fashioniq_pair_captions.py \
  --root /root/autodl-tmp/data/FashionIQ \
  --manifest /root/autodl-tmp/results/augmentation_retry/proposals.json \
  --output /root/autodl-tmp/results/augmentation_retry/qwen_reviewed.json \
  --model /root/autodl-tmp/src/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
  --limit 24
```

脚本默认检查现有 Shoes 的 `baseline.pid/queue.pid`，有存活进程即拒绝；此外拒绝任何已有 GPU 计算进程及不足 8 GiB 的磁盘余量。
模型和处理器强制 offline，不下载权重，不自动插入已有训练队列。OOM 会中止，已保存的响应和结果保留。

## 正式检索对比建议

先保留不扩充的 DQU 对照，再做 5%、10% 等经过独立复核的扩充量消融。
固定初始化、seed、优化器、batch、验证协议；同时控制优化步数/原始样本曝光次数，
避免“新增样本组训练步数更多”导致误判。扩充元数据可用于不同 seed 的重复实验，不能以验证标签反推描述。
报告同一原始验证集的 R@10、R@50，新增/拒绝样例数、重复率、审校方式及抽样错误率。
本次 0.334% 的小样本仅证明管线能用，不能据此判断方法有效。

相对描述合成方向可参考 [ConText-CIR（CVPR 2025）](https://openaccess.thecvf.com/content/CVPR2025/html/Xing_ConText-CIR_Learning_from_Concepts_in_Text_for_Composed_Image_Retrieval_CVPR_2025_paper.html)
和 [HyCIR](https://arxiv.org/abs/2407.05795)。本实现没有复现其完整训练/筛选方案。
