# 2026-10-19：迁移核验、FashionIQ 扩充与 CIG 试训

新服务器为 `connect.westc.seetacloud.com:42484`。凭据仅用于交互认证，不保存在代码、命令参数或 Git 中。旧服务器不再检查。

## Shoes 是否与 FashionIQ 一致

**六轮结构化分支／多正监督协议一致；没有重复 FashionIQ 此前的全部实验。**
相对 FashionIQ 多正实验提交 `5080813`，当前 `src/newModel.py` 和 `src/multi_positive_loss.py` 没有变化。
数据加载器增加了 Shoes 路径及字段适配。

| 项目 | FashionIQ Dress 多正实验 | Shoes |
|---|---|---|
| 结构化分支 | v7，字段掩码平均与有界残差 | 相同核心实现 |
| 训练 | seed42、batch16、lr5e-5、冻结 DQU | 相同 |
| 时长 | 2 轮 gate + 4 轮 adapter | 相同 |
| 精度／选模 | BF16 训练、FP16 检索、R@10+R@50 | 相同 |
| 难负项 | 权重 0 | 相同 |
| 训练／验证／验证图库 | 5985 / 2017 / 3817 | 8990 / 1761 / 4658 |
| 结构化有效验证覆盖 | 99.4051% | 83.0778% |
| annotated 额外不同 GT | 55 / 5985 | 64 / 8990 |
| identity | 目标 ID 全唯一，审计后跳过等价训练 | 实际训练；最终指标与 off 相同 |

两边均使用在线 Qwen3.7 Flash 的结构化文本，但领域提示词和验证过滤不同。Shoes 使用保守 v3 提示词、字段校验、同模型语义复核及颜色冲突拒绝；失败回退 DQU，不丢弃验证查询。它不是逐字复用 FashionIQ 的生成协议。
Shoes 未重复此前 FashionIQ v4–v8、三 seed、双写字图融合或伪目标图分支，不能说整个 FashionIQ 实验套件已经移植完成。

## Shoes 最终结果

继承已完成的实验，不重新训练基线或重新调用在线生成。五个后续阶段全部成功，三组分支各完成六轮。
train 有效 7488/8990（83.2925%），val 有效 1463/1761（83.0778%）；有效率不等于人工标注准确率。

| 方法 | R@1 | R@10 | R@50 | 相对本机 DQU 的百分点变化 |
|---|---:|---:|---:|---|
| 论文 DQU，五 seed 均值±std | 31.47±1.31 | 69.19±0.99 | 88.52±0.31 | 外部参照 |
| 本机 DQU，seed42 | 30.9483 | 68.7677 | 88.4725 | — |
| structured_off，最佳 epoch4 | 31.2323 | 68.8245 | 88.5292 | +0.2839 / +0.0568 / +0.0568 |
| structured_identity，最佳 epoch4 | 31.2323 | 68.8245 | 88.5292 | +0.2839 / +0.0568 / +0.0568 |
| structured_annotated，最佳 epoch3 | 31.0051 | 68.8813 | 88.4725 | +0.0568 / +0.1136 / 0 |

论文来源：[Table 2 与实现说明](https://arxiv.org/html/2404.15875v1#S4.T2)。同环境改进用本机 DQU 衡量，不能用论文五 seed 标准差给本机单 seed 推断显著性。当前没有多正监督稳定优于 off 的证据。
FashionIQ 六轮多正实验也未改善主指标：off 的 R@10+R@50 为127.8632，annotated 为127.6648。

## FashionIQ 继续扩充

代码快照：`/root/autodl-tmp/experiments/fashioniq-continuation-20261019`。
结果：`/root/autodl-tmp/results/fashioniq_augmentation_20261019`。

沿用之前筛查版 manifest 的全部144条记录（72接受、72拒绝），保留其内容和排除结果。本次另提议120个未尝试图对，seed44；不重新标注历史样本。累计 degree≤2、train-only、全类别 val/test 排除、噪声标记排除、相同图片字节与旧图对排除均保留。
使用原本地 Qwen3-VL-8B 权重、原双图生成和另一提示词复核，同一模型复核仍不是独立模型或人工 GT。
只新增元数据覆盖层，不增加图片或改动验证/测试。完成后另执行真实加载与抽样检查；尚无扩充后检索提升结论。

## CIG：用户新授权下的实模型试训

用户明确允许补齐权重，最多 **8 GPU 小时**，替代旧的禁止大权重下载约束。仍保留至少8 GiB数据盘空间。
作者仓库固定 `49d0be042ea0372c157844543dc80c4bef80fc9a`，本次复核公开目录仍只有两个 Phi 文件，没有可用训练后 U-Net；仓库提供推理与 Phi 训练，缺少扩散训练器。

因此实现独立的 **CIG-inspired SD1.5 FashionIQ 域内试训**：

- SD1.5 四通道文生图 U-Net + CLIP ViT-L/14 + 官方 SEARLE Phi。只选择需要的 safetensors/config，逐项验证来源 revision、大小与 SHA256，不下载整个仓库。
- 1024张原训练图片训练、另128张原训练图片作内部开发集；排除所有类别官方 val/test 的 ID 与相同图片字节，固定 manifest 后不重采样。
- 对每张图自身的 BLIP 描述删除第一个形容词／名词片段，以 `a photo of $ that …` 作为条件；参考图经冻结的 CLIP/Phi 得到伪词。重建对象是同一张训练图，无需 CIR 三元组标注。
- 冻结 Phi、CLIP、VAE 与 U-Net 其余参数，只训练 `attn2.to_k`、`attn2.to_v`；噪声预测 MSE，lr1e-6，microbatch2×累积8，有效batch16，最多500个优化步骤。
- 内部开发集固定噪声 seed 衡量重建 MSE；训练后最多8组固定内部开发样本做 base/trained 配对生成。这些指标不是检索准确率。
- 累计8小时墙钟从首次GPU使用开始覆盖缓存、训练、生成，持久化预算并设置硬截止；不自动重开预算。文件锁阻止重复启动。保留 best/last K/V 权重，不改 DQU。
- 等待本轮 FashionIQ 标注完成且GPU无其他计算进程后运行；不抢占标注GPU。依赖仅额外隔离安装 NLTK，不替换当前 Torch/Transformers/Diffusers。

与论文规模和部分细节有差别：作者约490k CC3M图文对、4×A6000；本试验使用 FashionIQ 域内图，标准中心裁剪、NLTK词性遮蔽、VAE后验mode缓存、单GPU混合精度。它既不是作者权重，也不是作者零样本结果复现。
论文生成图再次映射到伪词后的文本融合，与本项目将生成图经 DQU 目标编码器融合是不同评估方案。训练可运行不等于伪图可靠或检索提升。

代码：`src/cig_trial.py`、`scripts/run_cig_domain_trial.py`、`scripts/download_cig_trial_assets.py`。
CPU真实小模型测试覆盖 Diffusers K/V 更新、冻结参数不变、CLIP伪词注入及预算不能重置。
上游：[CIG](https://github.com/lan-lw/ComposedImageGen)、[论文](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Generative_Zero-Shot_Composed_Image_Retrieval_CVPR_2025_paper.pdf)、[SEARLE](https://github.com/miccunifi/SEARLE)。
