# 利用 Shoes 在线生成期间的 GPU 扩充 FashionIQ

2026-10-16 用户明确要求检查服务器 GPU，并提前继续 FashionIQ 三元组拓展。原先“等待 Shoes 全部结束”的安排据此调整为：FashionIQ 使用空闲 GPU，当前 Shoes API 生成继续，Shoes 的后续阶段调度等待本批标签任务释放 GPU。模型、候选图对和原定 120 对预算不变，不增加检索训练。

## 资源与实际启动

服务器报告 RTX 4080 SUPER，显存总量 32,760 MiB，检查时可用 32,230 MiB、无 GPU 计算进程，数据盘可用 21.64 GiB。足以载入已有 Qwen3-VL-8B-Instruct，无需下载权重。

核实并结束旧等待进程 3019 后，将原状态、日志、PID 和准备好的候选清单备份到：
`/root/autodl-tmp/results/fashioniq_augmentation_20261016/early_start_20261016`。
准备清单未重新抽样：120 pending＋20 accept＋4 历史 reject，总计 144；seed=43。

新代码目录 `/root/autodl-tmp/experiments/fashioniq-online-window-20261016`。
香港时间 **14:54** 启动监督进程 6822、双图标签子进程 6823，后者使用约 17.2 GiB 显存；这些 PID 仅是启动快照，后续应读取 `followup.pid` 并核实身份。

## 防止与 Shoes GPU 训练冲突

`cir_gpu_window.py` 只允许在 Shoes 的 `online-qwen37`、`qwen_train/qwen_val` 阶段取得 GPU 窗口。它核实调度父进程的 PID、Linux 启动时刻、命令哈希和 cwd，并只对该父进程 SIGSTOP；当前 API 生成子进程不暂停。父进程处于 `T` 是预约行为，不是进程崩溃。

窗口证明保存在 FashionIQ 结果目录的 `gpu_hold.json`，还绑定监督进程身份。双图生成器默认仍拒绝活跃 benchmark；只有明确提供有效证明，且调度进程确实被暂停、监督进程存活、GPU 无其他计算进程、磁盘至少 8 GiB 时，才允许在这个窗口内工作。不是删除原安全检查。

监督进程正常结束或异常退出时，先确保自己的标签子进程退出，再核实并 SIGCONT 原 Shoes 调度进程。此后 Shoes 按原队列继续。若当前 API split 恰好提前结束，切换下一阶段会等 FashionIQ 释放窗口，不会抢占 GPU。监督进程被强制杀死等异常须先查看证明、标签子进程和 GPU，再恢复同一调度进程，不能凭旧 PID 盲目信号。

## 扩充方法与边界

使用原 120 对提案，不替换类别、模型、提示词或采样 seed。只从同类别训练图片池配对，排除所有类别 val/test 图片 ID、已有图对、自身配对、完全重复图片和已标记噪声候选，绑定元数据及双图 SHA-256。

对每对图，使用两个已有训练图作为参考和目标，先生成两句相对描述及目标正属性关键词，再使用同权重、不同提示词双图复核。模型拒绝、格式不全或复核不确定的条目不导出。历史 20 条接受和 4 条拒绝原样保留；不强制放行、不重试被拒绝图对。

这里的双图输入是离线合成训练监督，不用于验证/测试查询文本生成。Shoes 的在线结构化文本仍只输入参考图和修改句。

新三元组只增加元数据。独立 overlay 的图片和 image_splits 指向原目录，原数据不改写；保留原 5,985 条训练标注前缀，追加新图对，并增加 DQU 所需的正属性关键词键。验证/测试标注不变。新图对尚无对应 query-only 结构化编辑，因此结构化加载器采用 DQU 回退；不能把它们宣称为已经训练过的结构化分支改进。

## 已验证的中间产物

在完整 120 对处理完成之前，另冻结了一份中间清单并构建 **preview**：本轮新接受 26 条＋历史 20 条，共 46 条新增；Dress 训练三元组 **5,985 → 6,031**。

```text
/root/autodl-tmp/results/fashioniq_augmentation_20261016/
  early_start_20261016/preview_manifest.json
  early_start_20261016/preview_overlay/
  early_start_20261016/preview_loader_check.json
```

这不是完整批次的最终数量。`verify_fashioniq_augmentation.py` 对该 preview 实际检查了：原标注前缀、训练图池 11,452 个 ID、共享图片目录、val/test 文件哈希不变；原 DQU 和结构化回退加载器均为训练 6,031、验证 2,017、图库 3,817。完整元数据构造期间仅替换图像 I/O 以避免堆积数千张图；另对每个加载器的 3 个新增图对实际解码并绘字，结果均为 512×512。没有载入检索模型权重或进行检索训练。

全部 120 对结束后，监督进程会导出正式 `overlay/`、`labeling_summary.json`、更新 `followup_state.json` 并释放 Shoes 调度。原始模型输出保存在 `qwen_reviewed.json.responses.json`。最终 overlay 应再次运行 CPU 验证：

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 python src/verify_fashioniq_augmentation.py \
  --root /root/autodl-tmp/data/FashionIQ \
  --overlay /root/autodl-tmp/results/fashioniq_augmentation_20261016/overlay \
  --output /root/autodl-tmp/results/fashioniq_augmentation_20261016/loader_check.json
```

## 质量观察和局限

助手视觉抽查了第 24–31 条双图及输出（不是独立人工 GT）：27 的黑色有肩带裙变银色抹胸、29 的黑色无袖变粉色长袖蕾丝，主要变化与图像相符。28 的花卉装饰可见，但“刺绣”这种具体工艺词仍需谨慎；31 的颜色/纹理变化成立，描述没有覆盖袖型的全部变化。

24–26 的图对均存在清楚衣物差异，模型却因“结构变化难以描述”拒绝，具有过度保守迹象。这些拒绝不能直接算作图片噪声，也没有据此删除原图片或强制接受。30 只生成一句且复核含不确定值，被格式/语义规则拒绝。

同权重的两次调用不是独立模型验证，更不是人工真值。当前接受数不能视为真实准确率；正式训练前仍需抽查具体工艺、否定、方向、未描述的结构变化等。此次仅证明扩充及加载管线能工作，尚无检索提升结果。

服务器 **84 项测试通过**，包括在线阶段准入、队列/监督进程身份、PID 重用拒绝、仍拒绝已有 GPU 工作和低磁盘、异常时恢复同一调度进程。持续监视已更新为并行模式，不能把有证明的 Shoes 暂停状态或 FashionIQ 占用 GPU 误报为冲突；完成后还须核实证明 `released` 及 Shoes 调度恢复。

## 追加视觉抽查与独立排除层

固定 seed=43，从抽样时已接受、索引至少 32 的新记录中抽取 6 对（34、70、47、99、82、42），连同前 8 对共检查 14 对；这不是全批准确率估计。34、47 的颜色及袖型/领型变化可见；70 的红色、袖子及交叠前襟变化可见，但深橄榄绿与近黑色的细分较难确认；82 的玫红及前短后长裙摆可见。

99 的目标仍有明显多层荷叶边，“更简单的圆领”具有误导性；42 的领口处在肩部上缘，“露肩”不够确定。连同 28 的“刺绣”工艺不确定，单独排除三个新记录。保留原始 `qwen_reviewed.json` 和主 `overlay/`，另在 `overlay_quality_screened/` 导出助手抽查后排除层，差异记录在 `quality_holdouts.json`。历史 20 条不变，其余未逐图审核的记录依然只是模型合成标注；该排除层也不构成人工 GT。原 120 对生成和自检协议未被修改，没有额外模型调用、重采样或检索训练。

## 完整批次结果

120 个新图对均完成：模型接受 **55**、拒绝 **65**，无 pending。加上历史 20 接受、4 拒绝，总清单为 accept=75、reject=69。原始 `overlay/` 将 Dress 训练三元组 **5,985 → 6,060**；抽查排除 28、42、99 后的 `overlay_quality_screened/` 保留本轮 **52** 条＋历史 20 条，共新增 **72** 条，训练 **5,985 → 6,057**。原始生成和模型自检接受率为 55/120=45.8%，不是真实准确率，也不代表检索效果。

扩充仅使用原图片。该类训练池仍为 11,452 个图片 ID；原 DQU 和结构化回退加载器的验证查询仍为 2,017、图库仍为 3,817。正式原始版和筛查版分别保留加载审计 JSON，构造全量元数据并各实际解码绘制 3 个新图对。筛查版可作为下一步训练输入；当前未训练，且多数接受记录尚未逐图复核。

监督任务完成后释放 GPU 窗口，记录 `released` / `same_dispatcher_resumed`，Shoes 调度按原在线生成和三分支训练协议接续。调度恢复及资源快照见 `completion_resource_check.json`。本轮没有增加在线 API 预算，没有新下载模型权重。
