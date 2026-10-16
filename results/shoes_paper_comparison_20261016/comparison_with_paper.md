# Shoes：论文与本机受控对比

状态：`complete`；完整对比：`True`。

数值为百分比；论文为五个 seed 的均值±标准差，本机为单 seed。

| 方法 | 状态 | R@1 | R@10 | R@50 | Avg |
|---|---|---:|---:|---:|---:|
| DQU-CIR paper | published_reference | 31.47 ± 1.31 | 69.19 ± 0.99 | 88.52 ± 0.31 | 63.06 ± 0.69 |
| DQU-CIR local | completed_baseline | 30.95 | 68.77 | 88.47 | 62.73 |
| structured_off | completed | 31.23 | 68.82 | 88.53 | 62.86 |
| structured_identity | completed | 31.23 | 68.82 | 88.53 | 62.86 |
| structured_annotated | completed | 31.01 | 68.88 | 88.47 | 62.79 |

论文：[Table 2](https://arxiv.org/html/2404.15875v1#S4.T2)；实验配置：[官方仓库](https://github.com/iLearn-Lab/SIGIR24-DQU-CIR)。

改进幅度以相同环境的本机 DQU 为基线，单位为百分点；论文行作为外部参考。单 seed 与论文五 seed 不可用于显著性结论。中间最佳 checkpoint 不等于完成实验。

原定三个结构化分支各六轮，冻结同一本机 DQU；保留全部验证查询和图库。额外训练预算尚未由匹配预算的继续训练对照排除。
