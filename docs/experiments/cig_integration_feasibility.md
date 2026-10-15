# 用 ComposedImageGen 替换 Qwen-Image 的可行性审计

检查日期：2026-10-15。对应此前聊天“评估 ComposedImageGen 集成方案”。
上游：[lan-lw/ComposedImageGen](https://github.com/lan-lw/ComposedImageGen)，固定提交 `49d0be042ea0372c157844543dc80c4bef80fc9a`。
论文：[Generative Zero-Shot Composed Image Retrieval, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Generative_Zero-Shot_Composed_Image_Retrieval_CVPR_2025_paper.pdf)。

**判断：生成器接口可以接入，值得小规模验证；当前公开资产不足以直接复现 CIG，不能认定它一定优于 Qwen 或能提升 DQU。** 本次仅审计源码、论文、权重目录及资源，不下载模型、不占用训练 GPU，没有生成 CIG 图片。

## 与现有项目的关系

| 环节 | 当前 Qwen-Image 路线 | CIG 路线 |
|---|---|---|
| 参考信息 | `generate_qwen_text_views.py` 只读取结构化目标描述；另有参考图条件的 Qwen-Image-Edit 路线 | 参考图经 CLIP 和 Phi 映射为伪词，与原始修改文本组成生成条件 |
| 生成器 | 大型 Qwen 图像模型；此前资源限制下测试 NF4/INT8 | 作者训练后的 SDXL UNet + 配套条件编码器 |
| 预期优点 | 文本语义表达灵活 | 相对纯文本生图，保留参考图信息的机制更直接；有机会使用较小 FP16 生成器 |
| 不能保证的性质 | 细节保真、修改准确、检索增益 | 同样不能保证；单个语义伪词不能锁定像素、纹理或鞋跟几何 |
| 项目中的接点 | 离线伪目标图及 manifest | 可以替换这一层，保留 DQU、结构化残差及真实图库 |

这替换的是 **Qwen-Image 生图**，不要求删除 Qwen3-VL 的结构化解析功能。
当前纯文本 Qwen 生图提示词固定为服装目录照，并不适合直接用于 CIRR 场景或 Shoes；若扩展生成实验，也必须修正这个领域假设。

当前记录中，Qwen-Image-Edit 的 20-query pilot 未通过打乱生成图对照；Qwen 纯文本生图 120-query pilot 的最佳主指标只增加约 0.10 个百分点，不能当作已确认的稳定改进。
这些是旧配置、旧 Dress checkpoint 的探索结果，不能与新 Shoes/CIRR 指标直接横向比较。原始记录见 `dress_qwen_structured_results.md`。

## 必须先解决的真实障碍

### 1. 关键训练后 UNet 未取得

作者 [README 权重链接](https://drive.google.com/drive/folders/1hpIpI0X26ox-uY-QdOPKDKKZlnWkftIA) 的当前公开目录只返回两个文件：

| 文件 | 目录记录的大小 |
|---|---:|
| `phi_best.pt` | 56,652,823 bytes |
| `phi_best_giga.pt` | 157,334,551 bytes |

未见 UNet 文件或子目录。这个结论仅针对本次可访问目录，不等于断言作者从未发布。
[`test_CIG.py:45`](https://github.com/lan-lw/ComposedImageGen/blob/49d0be042ea0372c157844543dc80c4bef80fc9a/test_CIG.py#L45) 明确加载 `model_path/unet` 的训练后模型，Phi 不能代替它。
仓库没有 CIG 扩散训练脚本；其中 `SEARLE_CIG/src/train_phi.py` 只训练文本反演网络，不能用于重建 CIG UNet。
因此，先取得匹配的训练后 UNet、模型配置、base 版本及 variant 信息，再谈复现实验。不能用普通 SDXL 生图并标为 CIG。

论文的方法包含对扩散模型交叉注意力 K/V 的专门训练，使用约 490K 图文对及多卡资源；在本服务器上重训不是一次轻量替换。
论文的实验包含 CIRR、FashionIQ、CIRCO、GeneCIS 和多个零样本基线，没有给出 DQU 或 Shoes 的直接证据，而且并非每个生成器/基线组合都提升。例如表 1 中 LinCIR ViT-G 的 CIRR R@1 在接入 CIG-XL 后反而降低。

### 2. 上游不是通用数据加载器

已核查 [`datasets/dataset_utils.py`](https://github.com/lan-lw/ComposedImageGen/blob/49d0be042ea0372c157844543dc80c4bef80fc9a/datasets/dataset_utils.py) 和 [`extract_lincir_feat.py`](https://github.com/lan-lw/ComposedImageGen/blob/49d0be042ea0372c157844543dc80c4bef80fc9a/extract_lincir_feat.py)：

- 特征提取器只接受 cirr/fashioniq，没有 Shoes。
- FashionIQ 返回项没有 `pairid`，但提取脚本保存时用 `pairid[idx]`，不能直接运行。
- CIRR 参考图片路径被写为 `root/split/reference.png`；不匹配官方 train 数字子目录以及 val 对应 dev 的布局。
- 生成脚本固定读取 CIRR test1 元数据，不能直接用于当前 train/val 对照。
- 输出文件只按编号命名，和下游五位补零文件名还可能不一致；按现有文件名跳过也不能验证旧图是否属于同一配置。

应使用本项目 `BenchmarkData` 的 CIRR/Shoes 路径与切分，FashionIQ 使用现有真实图片路径。
生成查询的 ID 应绑定 dataset、category、split、原始 query_index、reference 和修改文本哈希；不能仅凭图对合并不同指令。
生成输入清单只包含 query ID、参考图片和修改文本；真实 target ID 及标签留在独立评估侧，禁止传入生成器。

### 3. 条件空间与 DQU 不兼容

上游需要两套 CLIP/Phi，分别处理 768 和 1280 维条件；拼接条件为 `[77, 2048]`，另有 1280 维 pooled embedding。
DQU 的 ViT-H 图像特征是 1024 维，不能直接替换 Phi 的输入，也不能拿生成条件与图库特征点积。
先生成图片，再通过当前 DQU checkpoint 的 `extract_target()` 编码，才回到同一个检索空间。

### 4. 环境与磁盘不能照搬

上游锁定 Torch 2.0、Transformers 4.40、Diffusers 0.31；当前环境还需支持 Qwen3-VL。不要在训练环境执行上游全量 requirements 安装，应单独审查兼容性或隔离依赖。

本次通过 Hugging Face API 读取元数据，没有下载权重。CLIP-L 完整文件约 1.71 GB，CLIP-G 两个分片合计约 10.16 GB，Phi 合计约 0.21 GB。
再按 FP16 SDXL 规模估算 UNet、文本编码器和 VAE，常驻缓存预算就约 19 GB 十进制；实际 CIG UNet 大小仍未知。
服务器目前约 21.7 GiB 空闲，需保留 8 GiB 实验余量，不能直接套用完整缓存方案。
后续可在特征提取与生成阶段之间回收本次临时下载的独立编码器，或确认并回收已停止使用的图像生成模型；应先做实际文件预算，不删除现有有效基线及其依赖。

此外，上游每查询缓存 float32 条件约 0.6 MB；全量万级查询会额外占数 GB。先做 20/120 条，条件可在生成后删除，或在验证等价性后用 FP16 缓存。

## 建议接法与科学对照

```text
参考图 + 原始修改文本
    → 配套双 CLIP / Phi → 作者训练后 UNet → 伪目标图
    → 当前 DQU 的图像编码器 → g

DQU / 当前结构化查询 → q
q 与小权重 g 融合 → 原始真实图库排序
```

**先测简单版本**：`q' = normalize((1-w)*q + w*g)`，初始固定 `w=0.1`，不训练主干、不改变图库。
它是本项目适配方案，不是论文完整复现。现有 `evaluate_generated_image_views.py` 的 `--generated-encoding target --fusion-modes direct` 可供 FashionIQ 原型复用，但其输入绑定需加强，且不能直接用于 CIRR/Shoes。

**再测论文启发的语义融合**：论文及作者 SEARLE 实现会把生成图再次映射到伪词并结合修改文本，在文本空间融合。不能认为上一版建议的 image-only 融合就是作者完整算法。
DQU 已有 BLIP 描述和绘字查询，可另做“伪图图像特征 + 从伪图得到的绝对目标描述”分支；相对修改文本是否再次应用，需要作为单独消融，防止重复改变属性。
不要把 SEARLE 的 Phi 权重硬接进 DQU，也不要无对照地更换检索 backbone。

**可靠性门控**只用训练集学习，输入可包含修改满足度、保留属性、图像缺陷和参考信息保真度；先将不可信生成图权重置零。不能只看亮度或一项 CLIP 相似度，更不能用真实 target 相似度给验证样本择优。

建议执行次序：

1. 补齐训练后 UNet 和版本信息；做资源与离线加载检查。现阶段停在这里。
2. 当前训练队列结束后，在以前同一批 Dress 20-query 样本试生成，固定 seed、相同精度预算；检查衣物纹理、颜色、长度及 reference 属性保留。
3. 固定同一 checkpoint、图库和查询，比较原模型、Qwen 纯文本、Qwen-Image-Edit、CIG，以及打乱 CIG 图的负对照。保持生成图缺失时回退基线。
4. 主指标、样本平均排名和打乱对照都支持有用信号后，再做 120-query 确认。小子集结果与完整验证集指标分别报告，不把未覆盖查询造成的稀释误作“无效”。
5. 用 train/development 确定权重，锁定后评估保留数据和多个 seed；再扩展 Shoes、CIRR。不要继续在同一个 Dress validation pilot 上反复择优并声称最终泛化。

伪目标图起初只作查询的辅助视图，不加入真实图库、不直接标成多正样本真值。若未来用于训练蒸馏或伪正样本，需单独的可信度筛选和损失权重，防止放大生成错误。

## 监视状态

已为本聊天创建每 15 分钟的训练监视，普通进度保持安静，阶段完成、异常或明显新最佳结果才通知。需本地电脑和桌面应用保持运行。
本次最新已核验至 Shoes 第 4 轮评估，最佳仍为第 3 轮；第 5 轮训练正常。CIRR 仍受原始图片缺失阻塞。
本次没有改变训练模型、运行中代码快照、队列或权重，未部署 CIG。
