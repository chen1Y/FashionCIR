# 生成时尚图像描述（适配FashionBLIP-1模型）
import os
import json
import warnings
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm  # 进度条（可选，需安装：pip install tqdm）
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration # 适配自定义FashionBLIP-1

# 忽略无关警告（提升控制台整洁度）
warnings.filterwarnings("ignore")

# ===================== 配置参数（根据自己需求修改） =====================
IMAGE_FOLDER = "../data/FashionIQ/resized_image"  # 待处理的时尚图片文件夹路径
OUTPUT_JSON = "./fashion_image_descriptions_test.json"  # 时尚描述输出JSON
MODEL_NAME = "rcfg/FashionBLIP-1"  # FashionBLIP-1模型地址
USE_8BIT_QUANT = False  # 显存不足时设为True（需安装bitsandbytes）
# 🔥 FashionBLIP-1专属提示词（贴合时尚领域，提升描述准确性）
PROMPT_TEXT = "A detailed description of this image: "
# 生成配置（针对时尚描述优化，平衡细节与流畅度）
GENERATE_CONFIG = {
    "max_new_tokens": 150,  # 替换max_length（FashionBLIP-1推荐用max_new_tokens）
    "num_beams": 8,  # 增加beam数提升描述丰富度
    "repetition_penalty": 1.2,  # 降低重复惩罚（时尚描述易重复，适度调整）
    "temperature": 0.7,  # 降低温度提升描述稳定性
    "top_p": 0.9,  # 新增：核采样，提升时尚术语准确性
    "do_sample": True,  # 新增：启用采样生成更自然的描述
    "pad_token_id": 0,  # 适配FashionBLIP-1的pad token
    "eos_token_id": 100007,  # FashionBLIP-1专属eos token（避免截断）
}
SUPPORTED_FORMATS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# 🔥 测试模式：仅处理前N张图片
MAX_NUM_IMAGES = 10  # 👉 测试时修改，正式运行设为0

# ===================== 初始化FashionBLIP-1模型和处理器 =====================
def init_fashionblip_model():
    """初始化FashionBLIP-1模型（适配自定义时尚模型特性）"""
    # 设备自动选择（优先GPU，无GPU则CPU）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📱 使用设备：{device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")
    
    # 加载FashionBLIP-1专属处理器
    print(f"🔄 加载处理器：{MODEL_NAME}")
    try:
        processor = BlipProcessor.from_pretrained(
            MODEL_NAME,
            cache_dir="./fashionblip_cache",  # 独立缓存目录，避免与其他模型冲突
            trust_remote_code=True,  # 关键：FashionBLIP-1是自定义模型，需允许加载远程代码
            torch_dtype=torch.float16
        )
    except Exception as e:
        raise RuntimeError(f"❌ 处理器加载失败：{e}\n请检查模型名称是否正确，或网络是否能访问Hugging Face")

    # 模型加载参数（适配FashionBLIP-1）
    model_kwargs = {
        "torch_dtype": torch.float16,  # 半精度加载，降低显存占用
        "device_map": "auto",  # 自动分配设备（多GPU/CPU）
        "trust_remote_code": True,  # 必需：加载FashionBLIP-1的自定义代码
        "low_cpu_mem_usage": True,  # 降低CPU内存占用
    }
    
    # # 8bit量化（显存不足时启用，需安装bitsandbytes）
    # if USE_8BIT_QUANT and torch.cuda.is_available():
    #     model_kwargs["load_in_8bit"] = True
    #     model_kwargs["bnb_4bit_compute_dtype"] = torch.float16
    #     print("⚡ 启用8bit量化加载模型（降低显存占用约50%）")
    # elif USE_8BIT_QUANT and not torch.cuda.is_available():
    #     print("⚠️ 8bit量化仅支持GPU，自动禁用该功能")
    #     USE_8BIT_QUANT = False

    # 加载FashionBLIP-1模型
    print(f"🔄 加载FashionBLIP-1模型：{MODEL_NAME}")
    try:
        model = BlipForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            **model_kwargs,
            cache_dir="./fashionblip_cache"
        )
    except Exception as e:
        raise RuntimeError(f"❌ 模型加载失败：{e}\n解决方案：\n1. 确认网络能访问rcfg/FashionBLIP-1\n2. 升级transformers到4.30+版本\n3. 检查显存是否充足")

    # 模型预热（避免首次推理卡顿）
    if device == "cuda":
        model = model.eval()  # 推理模式
        print("✅ FashionBLIP-1模型初始化完成（已进入推理模式）")
    
    return processor, model, device

# ===================== 单张时尚图片生成描述 =====================
@torch.no_grad()  # 禁用梯度计算，节省显存
def generate_fashion_desc(image_path, processor, model, device):
    """生成单张时尚图片的专业描述（含完善的异常处理）"""
    try:
        # 加载并预处理时尚图片（适配服饰图片特性）
        with Image.open(image_path) as img:
            image = img.convert("RGB")
            # 额外预处理：确保图片尺寸合理（FashionBLIP-1推荐≥224x224）
            if min(image.size) < 224:
                image = image.resize((224, 224), Image.Resampling.LANCZOS)
        
        # 构建输入（融合提示词+图片）
        inputs = processor(
            images=image,
            text=PROMPT_TEXT,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device, torch.float16)

        # 生成时尚描述（使用优化后的配置）
        generated_ids = model.generate(
            **inputs,
            **GENERATE_CONFIG
        )
        
        # 解码生成结果（过滤特殊token）
        desc = processor.decode(
            generated_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        # 清理描述文本（移除提示词前缀，保留核心描述）
        if PROMPT_TEXT in desc:
            desc = desc.replace(PROMPT_TEXT, "").strip()
        
        # 过滤空描述
        if not desc or len(desc) < 5:
            return None
        
        return desc

    except UnidentifiedImageError:
        print(f"\n⚠️ 图片格式错误：{image_path}（无法识别为有效图片）")
        return None
    except torch.cuda.OutOfMemoryError:
        print(f"\n⚠️ 显存不足：{image_path}（建议启用8bit量化，或减少batch_size）")
        torch.cuda.empty_cache()  # 清空显存缓存
        return None
    except Exception as e:
        print(f"\n⚠️ 处理图片失败 {image_path}：{str(e)[:100]}")  # 截断过长错误信息
        return None

# ===================== 批量处理主函数 =====================
def batch_process_fashion_images():
    """批量处理时尚图片，生成专业描述"""
    # 初始化FashionBLIP-1模型
    try:
        processor, model, device = init_fashionblip_model()
    except RuntimeError as e:
        print(f"\n❌ 模型初始化失败：{e}")
        return

    # 收集所有有效图片路径
    image_paths = []
    for root, _, files in os.walk(IMAGE_FOLDER):  # 支持子文件夹递归查找
        for filename in files:
            if filename.lower().endswith(SUPPORTED_FORMATS):
                image_paths.append(os.path.join(root, filename))
    
    if not image_paths:
        print(f"❌ 文件夹 {IMAGE_FOLDER} 中未找到支持的图片文件（格式：{SUPPORTED_FORMATS}）")
        return

    # 测试模式：仅处理前N张图片
    original_count = len(image_paths)
    if MAX_NUM_IMAGES > 0:
        image_paths = image_paths[:MAX_NUM_IMAGES]
        print(f"\n📌 测试模式开启！仅处理前 {len(image_paths)} 张图片（总计找到 {original_count} 张时尚图片）")
    else:
        print(f"\n📌 正式模式：将处理全部 {original_count} 张时尚图片")

    # 批量处理（带进度条，实时显示进度）
    desc_dict = {}
    failed_paths = []
    for img_path in tqdm(image_paths, desc="处理时尚图片", ncols=100):
        # 获取图片唯一标识（保留完整路径的相对名，避免重名）
        rel_path = os.path.relpath(img_path, IMAGE_FOLDER)
        img_name = os.path.splitext(rel_path)[0].replace(os.sep, "_")  # 替换路径分隔符
        
        # 生成描述
        desc = generate_fashion_desc(img_path, processor, model, device)
        
        if desc:
            desc_dict[img_name] = desc
        else:
            failed_paths.append(img_path)

    # 保存结果到JSON（确保中文/特殊字符正常显示）
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)  # 自动创建输出目录
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(desc_dict, f, ensure_ascii=False, indent=4)

    # 输出详细统计信息
    total = len(image_paths)
    success = len(desc_dict)
    fail = len(failed_paths)
    print(f"\n✅ 处理完成！")
    print(f"📊 统计：总图片数={original_count} | 本次处理={total} | 成功生成={success} | 失败={fail}")
    print(f"📄 结果文件：{os.path.abspath(OUTPUT_JSON)}")
    
    # 输出失败列表（便于排查）
    if fail > 0:
        print(f"\n❌ 失败图片列表（前10个）：")
        for idx, path in enumerate(failed_paths[:10]):
            print(f"  {idx+1}. {path}")
        if len(failed_paths) > 10:
            print(f"  ... 还有 {len(failed_paths)-10} 个失败图片")

    # 提示正式运行的注意事项
    if MAX_NUM_IMAGES > 0:
        print(f"\n💡 测试完成！若效果满意，请：")
        print(f"   1. 将 MAX_NUM_IMAGES 设为 0 处理全部图片")
        print(f"   2. 修改 OUTPUT_JSON 为正式输出路径（如 ./fashion_image_descriptions.json）")

# ===================== 执行入口 =====================
if __name__ == "__main__":
    # 前置检查
    if not os.path.exists(IMAGE_FOLDER):
        print(f"❌ 图片文件夹不存在：{IMAGE_FOLDER}")
        print(f"   请检查路径是否正确，或确认文件夹已创建")
    else:
        # 检查依赖版本（确保兼容性）
        transformers_version = __import__("transformers").__version__
        if tuple(map(int, transformers_version.split("."))) < (4, 26, 0):
            print(f"⚠️ Transformers版本过低（当前：{transformers_version}）")
            print(f"   建议升级：pip install transformers>=4.30.2")
        
        # 执行批量处理
        batch_process_fashion_images()