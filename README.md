# FashionCIR 中文说明

FashionCIR 研究组合式图像检索。给定一张参考图和一句修改指令，系统从候选图库中找出符合修改要求的目标图。例如参考图是红色长袖衬衫，指令是“换成蓝色、保留长袖”，检索结果应尽量同时满足颜色变化和袖长保留。

## 从哪里读起

| 目录或文件 | 用途 |
|---|---|
| [src/](src/) | 数据读取、CLIP 编码、属性槽、损失函数、模型与检索评估。 |
| [scripts/](scripts/) | 数据准备、离线审计、单项实验和训练编排。运行前查看脚本参数及所需文件。 |
| [tests/](tests/) | 分割、排名、损失、数据隔离和模型模块的回归测试。 |
| [DQU_BASELINE.md](DQU_BASELINE.md) | 原生 DQU-CIR 的模型、输入与实验说明。 |
| [STRUCTURED_EDITS.md](STRUCTURED_EDITS.md) | 结构化文本生成、编辑字段和文本验证。 |
| [experiments/server_runs/](experiments/server_runs/) | 独立运行目录中与主工作树不同的源码，包括后续 034–042 的入口和依赖。 |
| [源码清单](docs/server_source_manifest.json) | 服务器原路径、发布路径和 SHA256。多份相同源码共用发布文件，清单保留每个来源。 |

## 主要实验路线

| 实验系列 | 做什么 | 入口或关键代码 |
|---|---|---|
| 原生 DQU-CIR | 修改词文本与写有关键词的参考图融合，作为整套系统的基线。 | [训练入口](src/train.py)、[模型](src/dquModel.py)、[数据](src/dquDataset.py) |
| 文本生成与数据检查 | 比较原始修改句、在线结构化编辑和图像编辑等路线。检查文本结构、重复图像和训练样本质量。 | [结构化生成](src/generate_structured_edits.py)、[数据审计](src/audit_fashioniq_noise.py) |
| 017–028 属性槽 | 将颜色、形状、细节、纹理编码成四个属性槽。包含属性对齐、门控和局部 Patch 匹配。 | [槽编码器](src/attr_slot_fusion.py)、[028 模型](src/fiq_gate_patch_model_028.py) |
| 029D／031 | 原修改句与属性槽动态混合，再与写字参考图做图文融合。031 对比输入原修改句与参考图 Caption＋纠错修改句。 | [029D 融合](src/fiq_fusion_029.py)、[031 入口](src/run_fiq_native_caption_031.py) |
| 034–037 | 依次考察 CLIP 全量解冻、负例集合、公开 CLIP 初始化，以及扩大到合规全训练集。 | 用下方恢复工具列出 `fiq-029d-*` 运行版本。 |
| 038A–D | 依次比较写字图、干净参考图、冲突擦除、视觉代理，另用 D 的同一权重验证图库 Patch 重排。 | 恢复 `fiq-pixel-free-038-full-exclusive`。 |
| 038E／038F | E 在 038C 上加全合法图库 KL。F 保留 038C 的损失，合并三类查询训练同一模型，分别按官方类别图库评估。 | 恢复 `fiq-pixel-free-038ef-gpu2-capacity`。 |
| 038G／039 | 联合训练加图库 KL。039A 只在参考图端使用 ViT-LoRA，039B 在参考图与图库共享 LoRA，并定期刷新图库特征。 | [LoRA 模块](src/fiq_vit_lora_039.py)、[039 入口](src/run_fiq_vit_lora_039.py) |
| 040／041／042 | 对比冲突感知文本、Qwen Flash 全量生成和 BLIP-2 描述输入。040H 记录 Flash 与原版 DQU-CIR 对照。 | [生成提示词](docs/experiments/fiq_context_generation_041_prompt.txt)、[040H 编排](scripts/run_fiq_040h_full_pair.py)、[042 入口](src/run_fiq_blip2_caption_generation_042.py) |
| 040 消融与重排 | 分别替换 DQU-CIR 的文本端和视觉端。额外验证属性擦除开关，并用 Flash 文本对原版 Top-100 候选做残差重排。 | [文本消融](scripts/run_fiq_040_text_ablation.py)、[图像消融](scripts/run_fiq_040_image_ablation.py)、[属性擦除验证](scripts/run_fiq_image_attribute_erasure.py)、[文本重排](scripts/run_fiq_flash_residual_rerank.py) |

## 环境与数据

服务器实际运行环境为 Linux、Python 3.12、PyTorch 2.9.0＋CUDA 12.6、torchvision 0.24.0、OpenCLIP 2.23.0。模型与运行脚本还使用 Pillow、NumPy、timm、Transformers、OpenCV、ftfy 和 regex。训练脚本使用 CUDA 与 `nvidia-smi`。旧实验的 CUDA 编译版本及可选生成依赖可能不同，优先看对应运行脚本和实验记录。

训练前自行准备有使用权限的 FashionIQ 或对应基准数据、图库图像、查询及官方划分。038 系列还需要 037 的字节隔离元数据、已归档的结构化属性文本和公开 OpenCLIP ViT-H/14 权重。对照统计需要 037 和 038C 的逐查询结果。完整实验计划 `plan.json` 按各自数据准备流程生成，记录源码、数据和对照结果的 SHA256。源码分支不分发训练数据、模型权重、运行日志或凭据。在线生成脚本需在自己的环境里配置 `DASHSCOPE_API_KEY`，038E／038F 使用已归档文本，无需在线调用。

历史脚本使用服务器当时的目录约定。换环境时先检查 `BASE`、数据根、Python 解释器、权重位置和运行目录，并使用目标环境的 GPU UUID。运行计划绑定源码与数据内容，修改路径配置后按实际准备流程重新生成对应计划。原始服务器任务的 Bash 编排可作为参考。

## 找到并恢复某次运行的源码

先列出已有快照，再恢复到单独的目录。恢复工具校验每份文件，遇到内容不同的现有文件会停止。

```bash
python scripts/restore_server_run_source.py --list
python scripts/restore_server_run_source.py \
  --run fiq-pixel-free-038ef-gpu2-capacity \
  --output ./local-038ef
export PYTHONPATH="$PWD/local-038ef/code:$PWD/src"
python local-038ef/code/run_fiq_pixel_free_038.py --help
```

使用已准备的 `plan.json`、数据和权重后，运行入口示例如下。`CUDA_VISIBLE_DEVICES`、`--gpu` 与运行计划必须指向同一个物理 GPU。E 分类别启动；F 一次传入全部三类。

```bash
export CUDA_VISIBLE_DEVICES="<GPU UUID>"
python local-038ef/code/run_fiq_pixel_free_038.py \
  --root ./local-038ef --category shirt --variant 038E \
  --gpu "$CUDA_VISIBLE_DEVICES"

python local-038ef/code/run_fiq_pixel_free_038.py \
  --root ./local-038ef --joint_categories dress,shirt,toptee --variant 038F \
  --gpu "$CUDA_VISIBLE_DEVICES"
```

038E 使用相同候选掩码对冻结目标图和组合查询的图库相似度做蒸馏，KL 权重为 0.5、温度为 0.07。038F 使用 16,242 条合规查询，类别内 batch 交错更新同一优化器，每批仍使用本类别负例图库。官方验证按 Dress、Shirt、Toptee 分别计算 R@1、R@10、R@50。运行目录的 `state.json` 记录阶段，`fit_results.json` 记录训练次数与权重校验值，`*_results.json` 保留逐查询排名和对照增损。

040H 的 `code/` 保存 Flash 训练依赖，`dqu_code/` 保存原版对照依赖。复现某次服务器运行时，先用 `--run fiq-040h-flash-full-vs-dqu-20261112` 恢复对应版本，再按入口脚本使用它们。040 文本消融有独立的 DQU 入口快照。图像消融、属性擦除验证和重排脚本接受既有运行目录，需另外准备权重、FashionIQ 图像和对应的文本或特征缓存。

## 本次源码同步的范围

主工作树源码放在原路径。独立运行和历史实验的不同版本通过源码清单索引，重复内容复用。备份包、Git bundle、图像、模型缓存、权重、结果日志和机器运行配置留在原存储。三个旧生成脚本的 Key 字面值改为读取环境变量；同时修复一个旧脚本的全角逗号语法错误，并匿名化一处历史主机标签。清单记录这些发布调整和相应校验值。
