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

### 本次完成结果

2026-10-19 15:37 香港时间，原120个新图对全部处理：模型接受51、拒绝69。原始覆盖层新增123（历史72+本轮51），总6108。
随后助手逐图检查全部51个新接受对，另外22条因明确矛盾或可视证据不足被单独排除，**本轮最终保留29条，累计新增101条，5985→6086**。
例如 index149 把明显有白领的参考衣物说成 collarless，index172 把有肩带的目标说成 strapless；其他排除项包括无法确认的面料工艺、领型及不可见参考背面。未把不确定条目直接改写成新真值。

推荐路径为服务器 `fashioniq_augmentation_20261019/overlay_quality_screened`，原始 `overlay` 和模型输出保留。历史144条 manifest 记录原样保留，历史72条接受数据没有重标。全部新接受样本的助手视觉检查也不是专家人工 GT。
两版均通过原 DQU 和 structured fallback 两个真实适配器加载：图片ID11452、验证2017、图库3817不变，原三元组前缀相同、val/test一致，并各实际渲染三个新图对。
可复核本地 `results/fashioniq_augmentation_20261019/quality_review_summary.json`、`quality_holdouts.json`、`screened_loader_check.json` 及两版 manifest。

## CIG：用户新授权下的实模型试训

用户明确允许补齐权重，最多 **8 GPU 小时**，替代旧的禁止大权重下载约束。仍保留至少8 GiB数据盘空间。
作者仓库固定 `49d0be042ea0372c157844543dc80c4bef80fc9a`，本次复核公开目录仍只有两个 Phi 文件，没有可用训练后 U-Net；仓库提供推理与 Phi 训练，缺少扩散训练器。

因此实现独立的 **CIG-inspired SD1.5 FashionIQ 域内试训**：

- SD1.5 四通道文生图 U-Net + CLIP ViT-L/14 + 官方 SEARLE Phi。只选择需要的 safetensors/config，逐项验证来源 revision、大小与 SHA256，不下载整个仓库。
- 1024张原训练图片训练、另128张原训练图片作内部开发集；排除所有类别官方 val/test 的 ID 与相同图片字节，并保守要求自身描述含 dress/gown，避免把不明确衣物类别混入本次 Dress 域内试训。固定 manifest 后不重采样。
- 对每张图自身的 BLIP 描述删除第一个形容词／名词片段，以 `a photo of $ that …` 作为条件；参考图经冻结的 CLIP/Phi 得到伪词。重建对象是同一张训练图，无需 CIR 三元组标注。
- 冻结 Phi、CLIP、VAE 与 U-Net 其余参数，只训练 `attn2.to_k`、`attn2.to_v`；噪声预测 MSE，lr1e-6，microbatch2×累积8，有效batch16，最多500个优化步骤。
- 内部开发集固定噪声 seed 衡量重建 MSE；训练后最多8组固定内部开发样本做 base/trained 配对生成，另对最多8个内部开发参考图的原训练查询做参考图＋修改句生成，生成器不读取目标图。这些指标不是检索准确率。
- 累计8小时墙钟从首次GPU使用开始覆盖缓存、训练、生成，持久化预算并设置硬截止；不自动重开预算。文件锁阻止重复启动。保留 best/last K/V 权重，不改 DQU。
- 等待本轮 FashionIQ 标注完成且GPU无其他计算进程后运行；不抢占标注GPU。依赖仅额外隔离安装 NLTK，不替换当前 Torch/Transformers/Diffusers。

与论文规模和部分细节有差别：作者约490k CC3M图文对、4×A6000；本试验使用 FashionIQ 域内图，标准中心裁剪、NLTK词性遮蔽、VAE后验mode缓存、单GPU混合精度。它既不是作者权重，也不是作者零样本结果复现。
论文生成图再次映射到伪词后的文本融合，与本项目将生成图经 DQU 目标编码器融合是不同评估方案。训练可运行不等于伪图可靠或检索提升。

代码：`src/cig_trial.py`、`scripts/run_cig_domain_trial.py`、`scripts/download_cig_trial_assets.py`。
CPU真实小模型测试覆盖 Diffusers K/V 更新、冻结参数不变、CLIP伪词注入及预算不能重置。
上游：[CIG](https://github.com/lan-lw/ComposedImageGen)、[论文](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Generative_Zero-Shot_Composed_Image_Retrieval_CVPR_2025_paper.pdf)、[SEARLE](https://github.com/miccunifi/SEARLE)。

### 启动核验（中间状态）

选定15个SD1.5/CLIP文件全部校验通过，另有官方SEARLE Phi（SHA256 `0ce31fb8eecaad95cdf60287dbad2ac9f968d79b74559f11b462701ce4693b95`）。
4项CPU测试通过，真实SD1.5 CLIP文本权重的原生伪词注入也通过，输出 `[1,77,768]`。
8小时累计预算始于 **2026-10-19 07:42:03 UTC（15:42香港时间）**。1152条缓存已完成，实模型训练已经开始，进程身份核实为5229，训练参数19,169,280个，仅K/V。
首步 checkpoint 已保存、参数变化L2为0.0037173；固定内部dev噪声MSE初始化0.0641924，首步0.0641843。此微小中间变化不能说明生成质量或检索收益。
本地 `results/cig_domain_trial_20261019/launch_verification.json` 是第52步时的启动快照，**不是最终结果**；服务器 `trial_state.json` 和 `training.jsonl` 才是继续推进的状态。
后续核验至第79步时GPU约4717MiB、无OOM/非有限值，磁盘约15.13GiB可用，不需要删除旧checkpoint。
已恢复每15分钟监视，正常进度不重复提醒；完成或失败后汇总并停止监视。没有启动扩充后的DQU检索训练。
