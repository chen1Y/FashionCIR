import argparse
import json
from pathlib import Path
import os

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
# 离线环境设置，禁止 Hugging Face Hub 的在线访问
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
def main():
    # 缺省的图片路径，建议运行时通过参数传入或直接修改
    default_image_path = "../data/FashionIQ/resized_image/B000AN2C1C.jpg"
    
    # 缺省的模型路径
    # 如果是本地下载，请修改为对应的本地绝对路径，例如: "/root/autodl-tmp/models--Qwen--Qwen3-VL-8B-Instruct"
    model_path = "Qwen/Qwen3-VL-8B-Instruct"
    cwd = os.getcwd()

    # 初始化配置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")

    # 加载 Qwen3-VL 的模型和处理器
    model = Qwen3VLForConditionalGeneration.from_pretrained("Qwen/Qwen3-VL-8B-Instruct", 
                                                            torch_dtype=torch.bfloat16,
                                                            cache_dir=cwd, 
                                                            attn_implementation="flash_attention_2",
                                                            local_files_only=True,
                                                            device_map="auto"
                                                            )
    processor = AutoProcessor.from_pretrained(model_path, cache_dir=cwd, local_files_only=True)

    # 需要推理的文本
    prompt = "请结合以下这段修改描述，详细描述这张图片修改后的服装内容，不要描述背景、人物等，一段话即可，以英文输出。修改描述：Is a brighter color and shorter in length,is yellow and shorter"

    try:
        # 打开图片
        image = Image.open(default_image_path).convert("RGB")
        print(f"Loaded image from: {default_image_path}")
    except Exception as e:
        print(f"读取图片失败，请检查路径是否正确: {default_image_path}")
        print(f"错误信息: {e}")
        return

    # 构建对话消息结构
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": default_image_path}, # 传入图片路径或已转换的内容
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # 添加 generation prompt 生成最终的文本
    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # 准备模型输入特征
    inputs = processor(
        text=[text_input],
        images=[image],
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    print("开始推理...")
    # 执行生成
    generated_ids = model.generate(**inputs, max_new_tokens=256)

    # 裁剪掉输入prompt部分，只保留生成的输出 token
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    # 解码生成的文本
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    print("\n[模型输出]:")
    print(output_text[0])

if __name__ == "__main__":
    main()



