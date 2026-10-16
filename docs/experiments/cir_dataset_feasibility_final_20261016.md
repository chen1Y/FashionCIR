# 后续 CIR 数据集：最终可行性核验

2026-10-16：Shoes 五阶段及 FashionIQ 扩充均已完成。本轮仅核验后续入口和评估协议，没有下载候选图片包，没有增加训练预算或启动新实验。

证据见 `results/shoes_paper_comparison_20261016/dataset_feasibility_final.json`。HEAD 成功只能说明端点和包大小可获取，不能当作完整数据下载、解压、CRC 或内容一致性验证。

| 候选 | 本次核验 | 与 DQU 接口及实验目标的关系 | 结论 |
|---|---|---|---|
| MIT-States | 官方 ZIP 直链 HEAD 200，807,745,643 bytes，约0.752 GiB | TIRG 参考实现有 noun-disjoint 切分、目标形容词及组合标签评估；原 DQU 固定版本没有 MITStates 类 | 首选适配候选；下载后须审计完整文件、解压体积和切分，另定义验证集与训练预算 |
| CSS3D | TIRG 官方链接仍在；本服务器访问 Drive 返回 Network is unreachable，完整二进制未确认 | 需要 `css_toy_dataset_novel2_small.dup.npy` 和 `images/*.png`；原 DQU 固定版本没有 CSSDataset 类 | 补充颜色/形状/大小组合测试，待解决获取问题，不能替代真实服饰实验 |
| CIRCO | COCO 图片 ZIP HEAD 200，20,126,613,414 bytes，约18.74 GiB | 官方只有220 val、800 test，采用多 GT 的 mAP@K；没有有监督 train | 当前21.49 GiB可用空间不足以同时容纳包、解压图片并保留8 GiB；暂不下载，不把 val 当 train |

[MIT 官方发布页](https://web.mit.edu/phillipi/Public/states_and_transformations/index.html)列出63,440张图片、245个名词与115个形容词。[TIRG 数据构造](https://github.com/google/tirg/blob/master/datasets.py)按固定测试名词划分，查询修改文本为目标形容词；其[评估](https://github.com/google/tirg/blob/master/test_retrieval.py)按目标“形容词＋名词”组合命中，而非唯一图片 ID。一个目标组合可以对应多张图片，这使其成为多正样本实验的合理候选；这是基于标签结构的可行性判断，不是实际检索效果。

当前项目的 `BenchmarkData` 只正式支持 CIRR/Shoes，FashionIQ 走已有专用加载器。原 [DQU 固定版本](https://github.com/iLearn-Lab/SIGIR24-DQU-CIR/blob/49f53992447a9a84b897d9b2e2f2c91dd2cc8788/src/datasets.py)只有 FashionIQ、Shoes、CIRR、Fashion200k；因此 MIT-States/CSS 需要新适配，不能称作原 DQU 论文已有的实验复现。MIT-States 应按组合标签构造合法多目标集合，不把同组合图片当负例，也不能用官方测试名词的数据选择超参数。

CSS 的输入及下载要求来自 [TIRG README](https://github.com/google/tirg#css3d-dataset)。CIRCO 的 split、GT 和完整图库要求来自[官方仓库](https://github.com/miccunifi/CIRCO)。CIRCO 的多 GT 标注适合未来评估，当前空间和有监督训练约束使它不是下一项直接训练选择。

本轮结束时保留全部有效权重；已完成 FashionIQ 筛查版72条合成扩充，但未训练，不能声称它改善检索。下一步先审计更丰富且可靠的正样本来源与标签质量，再安排明确预算的适配和受控实验。
