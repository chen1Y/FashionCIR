# 导入所需库
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# 1. 加载CLIPseg的处理器和模型（Hugging Face官方预训练模型）
# 处理器负责将图像/文本转换成模型能识别的格式
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
# 加载预训练模型
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

# 2. 准备输入（图像+文本提示词）
# 方式1：本地图像（替换成你的图像路径）
image_path = "../data/FashionIQ/resized_image/B00AU0N2SM.jpg"   # 比如一张包含"猫"的图片
image = Image.open(image_path).convert("RGB")

# 方式2：如果没有本地图，可使用示例图（需联网）
# from urllib.request import urlopen
# image_url = "https://example.com/your_image.jpg"
# image = Image.open(urlopen(image_url)).convert("RGB")

# 文本提示词（想要分割的目标，比如"猫"、"桌子"、"红色的车"）
text_prompts = ["dress"]  # 支持多提示词，比如["猫", "沙发"]

# 3. 预处理输入（图像+文本）
# processor会自动将图像缩放、归一化，文本转换成token
inputs = processor(
    text=text_prompts,
    images=[image] * len(text_prompts),  # 每个提示词对应同一张图
    return_tensors="pt",  # 返回PyTorch张量
    padding=True
)

# 4. 模型推理（无梯度计算，提升速度）
with torch.no_grad():
    outputs = model(**inputs)

# 5. 处理输出结果（分割掩码）
# outputs.logits是分割结果，形状为 [提示词数量, H, W]
seg_masks = outputs.logits  # 原始掩码（数值为得分，越大表示越匹配）

# 将掩码归一化到0-1区间（方便可视化）
seg_masks = torch.sigmoid(seg_masks)  # 先通过sigmoid转成概率
seg_masks = seg_masks.cpu().numpy()   # 转成numpy数组
print(seg_masks)
# 6. 可视化结果
plt.figure(figsize=(15, 5))

# 子图1：原始图像
plt.subplot(1, len(text_prompts)+1, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

# 子图2及以后：每个提示词对应的分割掩码
for i, (prompt, mask) in enumerate(zip(text_prompts, seg_masks)):
    plt.subplot(1, len(text_prompts)+1, i+2)
    # 显示掩码（用热图表示，越红表示匹配度越高）
    plt.imshow(mask, cmap="RdBu_r", vmin=0, vmax=1)
    plt.title(f"Segmentation:{prompt}")
    plt.axis("off")

# 保存/显示结果
plt.tight_layout()
plt.savefig("clipseg_result.png")  # 保存结果到本地
plt.show()