# Shoes 全量切换在线 Qwen3.7-Flash

2026-10-16 用户明确要求停止速度较慢的本地 Qwen3 生成，改用在线 Qwen3.7，生成后运行改进模型，与原 DQU-CIR 比较。这次是正式全量版本，独立于此前 24 条对照和 8 条事后回归检查；尚未产生新的检索结果。

## 已完成的切换

- 核实旧队列 PID 3018、生成子进程 3020 的命令行和 cwd 后停止，GPU 确认空闲；没有重训或删除原 DQU 权重。
- 保留本地 **1,452 条**生成记录。旧配置、日志、PID、JSON 和切换说明备份在 `/root/autodl-tmp/results/cirr_shoes_20261015/switch_to_online_20261016`。
- FashionIQ 等待进程 3019 在切换期间短暂冻结，根报告置为 `switching_to_online_qwen37` 后恢复等待；没有启动扩充，也没有改变其 120 条新图对＋20 条原样例安排。
- 新快照 `/root/autodl-tmp/experiments/shoes-online-full-20261016`，香港时间 **14:21** 启动后台队列，初始 PID **5853**、API 生成子进程 **5854**。后续必须读取 PID 文件并检查身份，不能依赖这些历史数字。

根结果仍为 `/root/autodl-tmp/results/cirr_shoes_20261015`，原基线 checkpoint 保留在其 `checkpoints/`。在线结构化文件另存：

```text
online_qwen37_v3/structured_shoes_train.json
online_qwen37_v3/structured_shoes_val.json
online_qwen37_v3/requests.sqlite3
```

`experiment_report.json.structured_paths` 是当前正式输入路径。根目录旧的 `structured_shoes_train.json` 保留为本地生成历史，不能用作在线进度或混入本次训练。

## 固定生成协议

全量重新处理 **8,990 train＋1,761 val**，不复用此前不同提示词的在线小样本，也不拼接本地模型输出。API 只收到参考图和修改句；目标 ID 仅保存为本地查询绑定，真实目标图片和目标描述不参与生成/自检。

模型固定 `qwen3.7-flash`，官方北京 workspace HTTPS 接口；非思考模式、temperature=0、JSON object，参考图最长边 512/JPEG90，生成最多 768 token、自检最多 256 token。版本 `shoes-online-v3-conservative-color-gate-20261016`，提示词和元数据指纹保存于配置及 SQLite 台账，续跑不得悄悄更改。

提示词沿用小样本的保守版本，再强调颜色替换后不保留受影响区域的旧颜色、含糊开口位置不强行具体化、复杂度分数含义和简短自检理由。无依赖 FashionIQ 的服装排除规则。

先做 schema 检查，再做透明的保守拒绝规则（完全相同的属性同时 retain/remove；明确最终配色与保留的鞋面颜色冲突），最后单独调用同一模型审阅。规则不静默改写输出。未通过的查询保留失败记录，检索回退 DQU；全部验证查询和图库均保留。

颜色规则可能保守误拒带其他色花纹的合理目标，自检也仍有误拒和漏检。因此通过数只表示可用结构化覆盖，不是人工准确率。原始生成、自检理由及失败原因均保存，不能将同一模型自检称为独立模型或人工 GT。

## 并发、费用与恢复

最多 **8 个线程、初始 120 请求/分钟**，429 时自动冷却并降低速率。单查询每阶段最多一次实际请求，**不自动重试**失败请求、语义不合格输出或未知返回；错误样例回退，连续传输失败或鉴权/模型错误则停止。API 生成不加载本地大模型，不占 GPU。

train/val **共用**一个 SQLite 台账，最多 **23,000 次请求、15 元标价预算**。按此前 8 条保守版本的实际 token 用量外推，10,751 条生成＋自检约 **8 元**，这是估算。
[官方北京价格](https://help.aliyun.com/zh/model-studio/qwen3-7-flash) 在单请求输入 ≤32k 时为输入 0.2、输出 0.8 元/百万 token。没有假设缓存或活动优惠。

请求发送前持锁原子预留 16,384 输入 token＋最大输出 token 的额度，已知 usage 返回后结算实际标价用量、释放未使用部分；未知/尚在途请求保留保守额度。`accounted_upper_cny` 是已结算＋保守预留，`estimated_list_price_cny` 是已返回 token 的标价估算。运行中的 `reserved` 和相应 `usage_missing_calls` 主要是在途请求，不能直接称为数据丢失或已扣费用。

SQLite 使用 WAL 和 FULL 同步，按 split＋原始 query index＋phase 唯一绑定，保存请求状态、usage、返回及最终样例。中断续跑复用已落盘响应和样例，失败也不重新生成；未知返回不盲目重发。预算与配置从原台账继承，不得续跑时重新获得 15 元额度。请求数、金额触顶或磁盘不足会停止后续实验，不能自动加预算。

密钥通过隐藏交互输入进入父进程，fork 后留在后台进程内存；每个生成子进程通过匿名管道读取。密钥不在 argv、环境变量、启动脚本、文件、日志或 Git 中。关闭 core dump，官方域名限制、TLS 验证、禁用代理与重定向，输出文本脱敏。SSH 断开不会终止已分离的队列；重启实例后内存密钥不会保留，不能假称自动恢复鉴权。

## 自动接续的检索比较

顺序保持：在线 train → 在线 val → `structured_off` → `structured_identity` → `structured_annotated`。

开始训练前，两个 split 必须处理完所有查询、绑定和生成版本一致，各自有效结构化覆盖至少 **50%**，并确认 GPU 无其他计算进程、数据盘至少剩 **8 GiB**。低覆盖不会悄悄进入训练。

三组都冻结同一个原 DQU-CIR 基线，seed=42，各 **6 epoch**，hard-negative 权重 0；数据切分、绘字处理、CLIP 权重、完整验证查询/图库、R@10＋R@50 选模规则不变。`off` 是关闭多正样本、仍使用结构化文本；另外两组按原先 identity/annotated 定义比较。

原 DQU-CIR 基线最佳第 3 轮：R@1 **30.9483**、R@10 **68.7677**、R@50 **88.4725**。队列最终汇总四组指标及相对基线差值，尚未训练完成不能将生成覆盖率说成检索提升。单 seed、额外 adapter 训练预算及环境差异仍限制结论强度。

后续用户已授权利用在线生成期间的空闲 GPU 提前扩充 FashionIQ，现已完成并释放 GPU；详见 [FashionIQ 并行扩充结果](fashioniq_online_window_20261016.md)。不再等待或重复启动该扩充。

## 验证与监视

服务器 **76 项测试通过**，包含并发费用/请求上限、真实用量释放预留、未知请求不重发、缓存与 split 绑定、协议/预算漂移拒绝、非连续原始索引、匿名密钥管道，以及假 API 的完整 train/val 导出＋重启续跑。假 API 测试不代表线上模型准确率。

14:24 左右首批 100 条中 91 条通过、9 条拒绝，已返回用量标价估算约 0.0716 元；这是启动快照，不是最终覆盖率。实时状态以服务器 JSON 和台账为准。

14:29 保存的快照为已处理 460 条、411 条通过、49 条失败，标价估算约 0.3189 元。实际 `CIRBenchmark` 加载线上部分产物成功，保留 8,990 条训练查询、1,761 条验证查询和 4,658 张图库；没有加载模型权重或删除验证查询。再次计算原 DQU checkpoint 的 SHA-256，与迁移时原始核验文件完全一致。切换记录、计划、源文件哈希和加载检查保存在 `results/shoes_online_full_20261016/`，只是启动证据，不是最终实验结果。

持续监视任务 `cir` 保持 ACTIVE，每 15 分钟检查在线路径、队列/子进程、API 错误与费用、生成推进、GPU/磁盘和后续 checkpoint，只在阶段完成、失败、明显新结果或需输入时通知。旧本地 Shoes 生成不得恢复。全部 Shoes 比较和 FashionIQ 扩充完成或停止且结果已报告后，再结束监视。


## 原论文结果与本机对照（2026-10-16 更新）

用户要求把原论文纳入对比。已核验作者 [arXiv 稿 Table 2](https://arxiv.org/html/2404.15875v1#S4.T2)：DQU-CIR 在 Shoes 上为 R@1=31.47±1.31、R@10=69.19±0.99、R@50=88.52±0.31，Avg=63.06±0.69，五个随机种子的均值±标准差。论文 4.1.2 使用单 A100-40G；[官方仓库](https://github.com/iLearn-Lab/SIGIR24-DQU-CIR) 环境为 Python3.9.18 / Torch2.0.1 / OpenCLIP2.23。

本机仍沿用现有单 seed42 和服务器报告的 RTX4080 SUPER / Torch2.9 / OpenCLIP2.23，不为追齐论文数字改动训练协议。原基线最佳 epoch3：R@1=30.9483、R@10=68.7677、R@50=88.4725，Avg=62.7295。与论文均值的差异分别为 -0.5217/-0.4223/-0.0475 个百分点，仅是描述性差异；不能从单 seed 与论文标准差推断显著性。改进幅度以相同环境、相同数据的本机基线为主。

只读报告工具 `scripts/report_shoes_comparison.py` 将论文、本机基线及三个结构化分支写入 JSON/Markdown，并记录各 R@k 相对本机基线的百分点差。不存在的结果保持 pending；已有但阶段未完成的最佳 checkpoint 标为 intermediate_best；只有原五阶段均成功且四组指标齐全才标完整对比。

服务器工具独立部署到 `/root/autodl-tmp/experiments/shoes-reporting-20261016`，不修改当前执行的 Shoes 源文件或调度进程，不新增训练或生成。监视时调用：

```bash
python /root/autodl-tmp/experiments/shoes-reporting-20261016/scripts/report_shoes_comparison.py \
  --results /root/autodl-tmp/results/cirr_shoes_20261015 \
  --output /root/autodl-tmp/results/cirr_shoes_20261015/comparison_with_paper
```

输出 `comparison_with_paper.json` 和 `.md`。最终仍需区分单 seed、额外 adapter 训练预算与方法本身的贡献；当前未增加匹配预算继续训练或多 seed 实验。


## 最终完成结果：香港时间 2026-10-16 18:31 核验

五个后续阶段均 returncode=0，三组各实际完成 6 轮，最终四组指标与日志和选模最大值一致。调度进程已退出，GPU 计算进程为空，数据盘仍有约 21.49 GiB。原 DQU checkpoint SHA-256 与在线切换前一致，训练源文件哈希均未变化；没有为了追齐论文改动协议。最终审计、六轮曲线、四个权重哈希和表格见 `results/shoes_paper_comparison_20261016/`。

| 方法 | 最佳 epoch | R@1 | R@10 | R@50 | 相对本机 DQU ΔR@1/10/50（百分点） |
|---|---:|---:|---:|---:|---|
| 原论文，5 seed 均值±std | — | 31.47±1.31 | 69.19±0.99 | 88.52±0.31 | 外部参考 |
| 本机 DQU，seed42 | 3 | 30.9483 | 68.7677 | 88.4725 | 0 / 0 / 0 |
| structured_off | 4 | 31.2323 | 68.8245 | 88.5292 | +0.2839 / +0.0568 / +0.0568 |
| structured_identity | 4 | 31.2323 | 68.8245 | 88.5292 | +0.2839 / +0.0568 / +0.0568 |
| structured_annotated | 3 | 31.0051 | 68.8813 | 88.4725 | +0.0568 / +0.1136 / +0.0000 |

这里按预先固定的 R@10+R@50 选模，不能把同一分支各轮最好的 R@1/10/50 拼成一行。三组最终选模分数同为 157.353776，本机 DQU 为 157.240204，只差 0.113572 个百分点的两项和。annotated 第4轮 R@1 虽为31.6298，但不是该协议选中的 checkpoint，不替换主表。

结果是小幅变化：结构化 off 稍高于本机基线，identity 的已选召回率与 off 相同，annotated 在 R@10 的微增伴随 R@1/R@50 低于 off，当前未展示稳定额外收益。单 seed、六轮预算、额外 adapter 训练以及模型自检标签限制了结论；不能宣称显著提升，不能由这次结果否定多正样本方法本身。

日志中的 distinct multi-positive query fraction：identity=0，表示没有新增不同目标 ID（仍会处理 batch 内同一目标重复列）；annotated=0.007119，即约64/8990查询，mean target bank size=16.11。它与 FashionIQ 的新增合成三元组不同，不是给所有 Shoes 查询自动补全真实多目标。覆盖较稀疏是可能限制因素，不是已由独立消融验证的因果解释。

在线生成已处理全部 8990 train 和1761 val：有效 7488 /1463，覆盖83.2925% /83.0778%；其余1502 /298保留并回退 DQU，未删除验证查询或改变4658张图库。格式/保守规则和同模型自检通过仍不是人工准确率。总20805次请求，20803有用量、2次未知返回保持保守预留，无在途请求；标价估算7.651109元，含未知预留的上界7.658072元，未达到15元或23000次上限，不将它称为实际账单。

FashionIQ 已独立完成72条筛查后扩充（本轮52+历史20，5985→6057），无新增图片、无检索训练，因此没有其召回提升结论。MIT-States/CSS 后续候选见本次最终可行性记录，不自动增加训练预算或下载大权重。全部结果报告后关闭本轮监视。

后续数据集的最终取舍及官方来源见 [MIT-States / CSS / CIRCO 可行性核验](cir_dataset_feasibility_final_20261016.md)。
