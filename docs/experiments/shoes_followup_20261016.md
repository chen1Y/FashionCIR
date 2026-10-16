# Shoes 监视及后续任务（2026-10-16）

**最终更新：** Shoes 五阶段现已全部完成，FashionIQ 已按最新授权提前扩充完成；下文等待安排与早期 PID 是历史快照，不再执行。最终结果见 [Shoes 全量在线结果](shoes_online_full_20261016.md#最终完成结果香港时间-2026-10-16-1831-核验)及 [FashionIQ 扩充结果](fashioniq_online_window_20261016.md)。

## 当前状态，香港时间 01:00

Shoes 尚未完成全部实验。原 DQU 在第 8 轮正常早停，以第 3 轮的 R@10+R@50 选中模型：
R@1=30.9483，R@10=68.7677，R@50=88.4725，合计 157.2402。
这些是已完成的基线指标，结构化分支的最终对比仍未产出。

当前为 `qwen_train`，已处理 373/8,990 个训练查询，215 条通过、158 条未通过解析或语义自检。
自检是同模型质量信号，不是人工真值；此前发现部分自检将编辑后的目标状态与参考图直接比较而误拒，
风险已记录，未修改运行中的提示词或实验协议。失败样例按原方案回退 DQU，最终必须报告有效覆盖率。
Shoes source_index 来自训练/评估混合的原始 JSON，因此训练进度不能用最后的 source_index+1 代替。

已核实 queue PID 2996、Qwen PID 4561，输出仍推进；没有在记录中发现 OOM、CUDA error 或 NaN。
数据盘可用 21.73 GiB；未删除有效 checkpoint。进程 PID 只代表本次快照，后续必须核对命令行，不能盲目信任旧 PID。
快照见 `results/cirr_shoes_20261016/monitor_20261016.json`。

## 用户指定的后续优先级

1. 原队列完整执行：Shoes 基线 → Qwen train/val → structured off/identity/annotated。
2. 所有阶段成功、最终四组指标齐全有效、原进程退出且 GPU 空闲后，先扩充 FashionIQ Dress 三元组。
3. 之后再考虑容易公开获取的其他 CIR 数据集。当前只核实入口和协议，没有下载大包或启动新数据集训练。

监视任务 `cir` 已恢复 ACTIVE，每 15 分钟检查一次，仅在阶段完成、失败、明显新结果/质量异常或需用户处理时通知。
它保留当前聊天上下文。桌面定时检查需要 Codex 及本机可运行；下面的服务器等待任务不依赖聊天窗口保持打开。
两部分均有状态记录，不能重复启动 GPU 任务。官方任务说明见
[OpenAI Docs / Scheduled tasks](https://learn.chatgpt.com/docs/automations?surface=app)。

## 服务器 FashionIQ 等待任务

- 代码：`/root/autodl-tmp/experiments/fashioniq-augmentation-followup-20261016`。
- 输出：`/root/autodl-tmp/results/fashioniq_augmentation_20261016`。
- 状态：`followup_state.json`，日志：`followup.log` / `labeling.log`，PID：`followup.pid`。
- 本次已核实 PID 7147，状态 `waiting_for_shoes`；目前只有 CPU 等待，没有加载第二个模型。
- 预生成清单 `prepared_proposals.json` 包含此前 20 条接受样例、4 条历史拒绝和 **120 条新的 pending 图对**。
- 新候选 seed=43，同类、train-only、全类别 held-out ID 排除、原噪声审计采样排除；限制图片新增边数并排除历史图对。
- 既有 20 条接受样例保持不变。新增标注使用现有本地 Qwen3-VL-8B，双图生成后重新调用同一模型复核，
  显式要求两张图均属于 Dress，保留原始响应，不将同权重复核称为独立模型或人工审校。
- 预算是 **120 次新图对尝试**，不是保证产生 120 条合格数据。最终仍使用原 11,452 个 Dress 训练图片 ID，
  训练三元组数为 5,985+20+新增通过数。单独导出覆盖层，原始图片和标注不覆盖。
- 只扩充标注，没有新增检索训练预算，也不自动声称召回率提升。

`scripts/queue_fashioniq_augmentation.py` 用 Linux 文件锁阻止重复运行。只有五个后续阶段均 returncode=0，
四组 R@1/10/50 齐全、有限且在 0..100 范围，才接受 Shoes 成功证据。
Shoes 失败/停止、旧 PID 身份变化、空间低于 8 GiB、模型/生成失败、源数据指纹变化均停止后续流程并保留输出。
不会自动重试部分生成结果或覆写日志，必要时先查明原因。全部 52 项服务器测试已通过；真实新双图推理尚未开始。
扩充结束后的加载验证和抽样视觉检查仍须完成，不能把等待状态或模型自检直接当成新增数据质量结论。

## 其他数据集：当前可获取性和适用性

仅做 HEAD 请求和官方协议检查；没有下载以下数据包。入口探测见 `dataset_download_probes.json`。

| 候选 | 实测获取情况 | 当前取舍 |
|---|---|---|
| MIT-States | 官方 HTTP 直链在本机和服务器均 200；807,745,643 bytes，约 0.752 GiB ZIP | **后续首选候选**；下载前仍需核实解压体积、CRC 和适配协议 |
| CSS | TIRG 给出 Drive 链接；目前返回确认 HTML，未验证实际二进制下载 | 作为颜色/形状/大小组合能力的补充测试，不能代替真实服饰检索 |
| CIRCO | 官方标注公开，COCO unlabeled2017 直链在服务器 200；20,126,613,414 bytes，约 18.74 GiB ZIP | 单压缩包就会破坏 8 GiB 余量，且还需解压；暂不下载。只有 val/test，无有监督 train |
| i-CIR | 官方 HTTPS 直链 200；61,454,288,650 bytes，约 57.23 GiB 压缩包 | 超出当前空间，暂不下载 |

MIT 官方入口为 [States and Transformations](https://web.mit.edu/phillipi/Public/states_and_transformations/index.html)。
官方发布页使用 HTTP 数据链接；自动升级 HTTPS 的入口证书主机名不匹配，因此本次使用发布页原始 HTTP 链接进行 HEAD 检查，未关闭 TLS 验证。
其 [TIRG 数据构造](https://github.com/google/tirg/blob/master/datasets.py) 按名词划分训练/测试，
修改文本是目标形容词；[官方评估](https://github.com/google/tirg/blob/master/test_retrieval.py) 按目标“形容词+名词”组合判断命中，
不能照搬 FashionIQ 单目标 ID 评估或把同组合其他图片当负例。现有新训练入口尚未正式适配 MIT-States，
要先确定输入构造、完整评估和对照，再进行小试；不能称为原 DQU 论文已有的 MIT 结果。

参考：[TIRG / CSS](https://github.com/google/tirg#css3d-dataset)、
[CIRCO 官方仓库](https://github.com/miccunifi/CIRCO)、[i-CIR 官方仓库](https://github.com/billpsomas/icir)。
CIRCO 可以将来做跨数据集、多目标评估，但不能拿 220 条验证标注充当有监督训练集，
也不能只下载命中图片后将小图库成绩冒充官方全库指标。
