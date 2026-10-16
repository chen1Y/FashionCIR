# 2026-10-16 新克隆实例恢复

新服务器：`connect.weste.seetacloud.com:42002`。不在文件中保存 SSH 密码。
分支：`codex/resume-cloned-server`。

## 昨日结果与克隆实际保留内容

- Shoes 原 DQU 已正常完成，第 8 轮早停，按 R@10+R@50 选第 3 轮。
  R@1=30.9483、R@10=68.7677、R@50=88.4725，合计 157.2402。
- 克隆保留的训练集结构化记录有 **1,096/8,990** 条，其中 598 条通过模型自检、498 条失败或被拒绝。
  它们比旧监视最后可连接时的 424 条更多，故不能将旧监视快照当作克隆实际进度。
  失败包含 schema/JSON 不合规和语义自检拒绝，不是进程崩溃或人工质量标签。
- Shoes 验证集结构化生成及 off/identity/annotated 三个训练分支尚未开始，没有最终方法对比或提升结论。
- FashionIQ Dress 已保留 20 条经过 Codex 双图检查的合成伪标注，训练三元组 5,985→6,005，训练图片 ID 仍为 11,452。
  下一批 120 个新图对已经准备，仍需等 Shoes 全部完成后才启动；尚未进行扩充后的检索训练。
- MIT-States 是后续公开数据集候选；此前只核验了官方链接与协议，没有下载并启动新数据集训练。

## 恢复前验证

新实例 GPU 无计算进程，旧 queue / baseline / followup 进程均不存在。
数据盘剩余约 21.67 GiB，足以继续当前队列，未删除有效权重或下载大权重。
Shoes 路径审计无缺失图片，训练/评估查询仍为 8,990 / 1,761，图片池仍为 10,000 / 4,658。
生成输入 query SHA256 与旧记录一致。

基线 checkpoint 约 3.96 GB，可完整加载，epoch=3，dataset=shoes，693 个模型张量均通过有限值检查。
SHA256：`0f5972dc791caa78b713dcbac60b883e22c51c7d27d3d6b2de88812985cfaaf2a`。
55 项服务器测试通过。详见 `results/cirr_shoes_recovery_20261016/recovery_preflight.json`。

## 断点修复与保持的协议

旧生成器续跑时只跳过 `ok`，会重新尝试保存为 `failed` 的样例。
为了继续原有“每查询一次尝试”的预算，新 `resume_samples` 同时保留并跳过所有已完成的 `ok/failed` 记录，
拒绝重复索引、未知索引、错误状态或与查询不一致的旧记录。原 498 条失败不会悄悄获得额外生成机会。
增加 attempted/remaining/attempts_complete 元数据，明确“全部尝试完”与“全部生成有效”的区别。

队列日志改为追加写入并标记新启动时间。旧日志、生成 JSON、实验配置、旧 PID 文件、FashionIQ 等待状态和候选清单已备份到：
`/root/autodl-tmp/results/cirr_shoes_20261015/recovery_20261016_new_instance`。
只移除已核实无存活进程的旧 baseline PID 标记，新 PID 不沿用克隆前的数字。

恢复代码位于 `/root/autodl-tmp/experiments/cirr-shoes-recovery-20261016`，原快照保留。
与原 Shoes 快照共有的 src 文件中，仅生成器断点处理不同；生成 prompt、critic、图片处理函数经 AST 比较完全一致。
模型、本地权重、查询、每查询生成预算、baseline、seed=42、三个分支各 6 轮和评估协议均保持一致。
原有自检误拒风险继续记录，未在恢复时修改提示词或重标旧样例。

## 已实际恢复的队列

香港时间 **12:49** 重新启动；12:52 检查时：

- Shoes queue PID 3018，生成子进程 PID 3020，进程身份和工作目录已核实。
- 只生成未处理的 7,894 条 train，之后生成 1,761 条 val，再依次训练 off / identity / annotated。
- 保存记录增至 **1,105**，新增 9 条；原有 1,096 条记录对象完全一致。
- 旧 qwen_train / queue / baseline 日志的字节前缀完整保留，基线没有重训。
- GPU 上只有 Shoes 生成进程，使用约 17,252 MiB 显存，无 OOM/CUDA error/NaN。
- FashionIQ 等待任务 PID 3019，状态 `waiting_for_shoes`，仍保留 120 次新图对尝试及此前 20 条接受样例。

进程 PID 仅是此次快照，后续需读取 PID 文件并核对命令行。
恢复验证见 `results/cirr_shoes_recovery_20261016/recovery_verification.json`。
等待任务仍从原 FashionIQ 独立目录运行；其模型只有在 Shoes 全部成功结束且 GPU 空闲后才会加载。

监视任务 `cir` 已更新新地址，ACTIVE，每 15 分钟运行；新状态保存在本地 `.autodl/training_monitor_state.json`，
旧网络故障状态已清除，后续只在阶段完成、明显新结果、失败或需处理时通知。
原来的“全部实验结束或停止且已报告后关闭监视”规则保留。
