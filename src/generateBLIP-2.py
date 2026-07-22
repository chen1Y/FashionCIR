import os
import json
from PIL import Image
from tqdm import tqdm  # 进度条（可选，需安装：pip install tqdm）
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# ===================== 配置参数（根据自己需求修改） =====================
IMAGE_FOLDER = "../data/FashionIQ/resized_image"  # 待处理的图片文件夹路径（绝对/相对路径）
OUTPUT_JSON = "./image_descriptions_test.json"  # 测试用输出JSON（区分正式结果）
MODEL_NAME = "Salesforce/blip2-opt-2.7b"  # BLIP-2模型版本（平衡效果与显存）
USE_8BIT_QUANT = False  # 显存不足时设为True（8bit量化，显存减少50%）
PROMT_TEXT = ""
# 生成描述的参数（参考前文说明）
GENERATE_CONFIG = {
    "max_length": 150,
    "num_beams": 4,
    "repetition_penalty": 1.2,
    "temperature": 0.7
}
SUPPORTED_FORMATS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# 🔥 测试模式：仅处理前N张图片（重点修改这里！）
MAX_NUM_IMAGES = 10  # 👉 想要测试多少张就改这个数（设为0则处理全部）

# ===================== 初始化模型和处理器 =====================
def init_blip2_model():
    """初始化BLIP-2模型和处理器"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备：{device}")
    
    # 加载处理器
    processor = Blip2Processor.from_pretrained(MODEL_NAME,cache_dir="./")
    
    # 加载模型（根据量化配置）
    model_kwargs = {
        "torch_dtype": torch.float16,
        "device_map": "auto"  # 自动分配设备（多GPU/CPU）
    }
    if USE_8BIT_QUANT and torch.cuda.is_available():
        model_kwargs["load_in_8bit"] = True
        print("启用8bit量化加载模型（降低显存占用）")
    
    model = Blip2ForConditionalGeneration.from_pretrained(MODEL_NAME, **model_kwargs, cache_dir="./")
    return processor, model, device

# ===================== 单张图片生成描述 =====================
def generate_image_desc(image_path, processor, model, device):
    """生成单张图片的描述（含异常处理）"""
    try:
        # 加载并预处理图片
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, text=PROMT_TEXT, return_tensors="pt").to(device, torch.float16)
        
        # 生成描述
        generated_ids = model.generate(**inputs, **GENERATE_CONFIG)
        desc = processor.decode(generated_ids[0], skip_special_tokens=True).strip()
        return desc
    
    except Exception as e:
        print(f"\n⚠️ 处理图片失败 {image_path}：{str(e)}")
        return None

# ===================== 批量处理主函数 =====================
def batch_process_images():
    # 初始化模型
    processor, model, device = init_blip2_model()
    
    # 收集所有图片路径
    image_paths = []
    for filename in os.listdir(IMAGE_FOLDER):
        if filename.lower().endswith(SUPPORTED_FORMATS):
            image_paths.append(os.path.join(IMAGE_FOLDER, filename))
    
    if not image_paths:
        print(f"❌ 文件夹 {IMAGE_FOLDER} 中未找到支持的图片文件")
        return
    
    # 🚨 测试模式：仅处理前N张图片
    if MAX_NUM_IMAGES > 0:
        original_count = len(image_paths)
        image_paths = image_paths[:MAX_NUM_IMAGES]  # 截取前N张
        print(f"\n📌 测试模式开启！仅处理前 {len(image_paths)} 张图片（总计找到 {original_count} 张）")
    else:
        print(f"\n📌 正式模式：将处理全部 {len(image_paths)} 张图片")
    
    # 批量处理图片（带进度条）
    desc_dict = {}
    for img_path in tqdm(image_paths, desc="处理图片（测试效果）"):
        img_basename = os.path.basename(img_path)
        img_name = os.path.splitext(img_basename)[0] # 保留图片原名（不含扩展名）
        desc = generate_image_desc(img_path, processor, model, device)
        if desc:
            desc_dict[img_name] = desc
    
    # 保存到JSON文件
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(desc_dict, f, ensure_ascii=False, indent=4)  # indent=4 格式化输出
    
    # 输出统计信息
    total = len(image_paths)
    success = len(desc_dict)
    print(f"\n✅ 测试处理完成！")
    print(f"📊 本次处理图片：{total} | 成功生成：{success} | 失败：{total-success}")
    print(f"📄 测试结果已保存至：{os.path.abspath(OUTPUT_JSON)}")
    print(f"\n💡 若效果满意，可将 MAX_NUM_IMAGES 设为 0 处理全部图片，并修改 OUTPUT_JSON 为正式路径")

# ===================== 执行主函数 =====================
if __name__ == "__main__":
    # 检查文件夹是否存在
    if not os.path.exists(IMAGE_FOLDER):
        print(f"❌ 图片文件夹 {IMAGE_FOLDER} 不存在，请检查路径！")
    else:
        batch_process_images()