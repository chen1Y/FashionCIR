import os
import json
import base64
from openai import OpenAI


def img2base64(img_path):
    """将图片转换为base64编码字符串"""
    with open(img_path, "rb") as image_file:
        base64_str = base64.b64encode(image_file.read()).decode("utf-8")
    return base64_str


# 初始化客户端
client = OpenAI(
    api_key=os.environ["DASHSCOPE_API_KEY"],  # 替换为你的API Key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 配置路径
img_dir = "../data/FashionIQ/resized_image/"
# caption_file = "../data/FashionIQ/captions/modified_captions_train.json"
caption_file = "../data/FashionIQ/captions/cap.dress.val.json"
img_caption_file = "../data/FashionIQ/captions/image_captions_dress_val.json"
output_file = "modified_val_test.json"

# 控制处理的数据量（先试10条，可调整）
sample_size = 10

# 读取原始数据
with open(caption_file, 'r', encoding='utf-8') as f:
    all_data = json.load(f)
with open(img_caption_file, 'r', encoding='utf-8') as f:
    caption_data = json.load(f)

# 只取部分数据（前sample_size条）
sample_data = all_data
# sample_data = all_data[:sample_size]
results = []

# 遍历样本数据
for item in sample_data:
    candidate_id = item["candidate"]
    captions = item["captions"]  # 假设包含至少2条修改指令
    # captions = item['modified_texts']
    original_target = item["target"] # 保留原始target
    caption_img = caption_data[candidate_id]
    
    # # 构建图片路径
    # img_path = os.path.join(img_dir, f"{candidate_id}.jpg")
    # if not os.path.exists(img_path):
    #     print(f"警告：图片 {img_path} 不存在，跳过")
    #     continue
    
    try:
        # base64_image = img2base64(img_path)
        # 存储当前candidate的两条结果
        modified_results = {
            "candidate": candidate_id,
            "target": original_target,  # 保留原始target
            "modified_texts":""  # 存放两条修改后的文本
        }
        modify_caption = captions[0] + " and " + captions[1]
        # modify_caption = captions
        # 调用模型生成修改后的描述
        completion = client.chat.completions.create(
            model="qwen-flash",
            stream=False,
            messages=[{
                "role": "user",
                "content": [
                    # {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    {"type": "text", 
                             "text": f"I have an image. Given an instruction to edit the image,carefully generate a description of the edited image. I will put my image content beginning with ’Image Content:’.The instruction I provide will begin with ’Instruction:’. Each time generate one edited description only.不要生成文中没有的东西，生成的文本中不要出现矛盾。保留原描述中未被修改的部分，突出修改的部分。注意instruction都是对服装的描写而非人。直接输出结果不需要带前缀。"
                                      f"Image Content：{caption_img}"
                                      f"Instruction:{modify_caption}"}
                    ]
                }]
            )
        # 提取结果并添加到列表
        modified_text = completion.choices[0].message.content
        modified_results["modified_texts"] = modified_text
        # 将当前candidate的结果加入总列表
        results.append(modified_results)
        print(f"已处理 {candidate_id}（{len(results)}/{sample_size}）")
        
    except Exception as e:
        print(f"处理 {candidate_id} 出错：{str(e)}")
        continue

# 保存结果
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"部分数据处理完成，结果已保存到 {output_file}（共{len(results)}条有效数据）")




























# import os
# import json
# import base64
# from openai import OpenAI


# def img2base64(img_path):
#     with open(img_path, "rb") as image_file:
#      # 编码为 Base64 字符串（去掉前缀，仅保留编码部分）
#         base64_str = base64.b64encode(image_file.read()).decode("utf-8")
#     return base64_str

# img_path = "../data/FashionIQ/resized_image/"
# output_file = "modified_descriptions.json"
# with open("../data/FashionIQ/captions/cap.dress.train.json",'r',encoding='utf-8') as f:
#     data = json.load(f)
# results = []
# client = OpenAI(
#     # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key=os.environ["DASHSCOPE_API_KEY"],
#     api_key=os.environ["DASHSCOPE_API_KEY"],
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
# )
# image_url = ""
# captions = ""
# for item in data:
#     candidate_id = item["candidate"]
#     img = os.path.join(img_path, candidate_id)
#     img = img + ".jpg"
#     image_url = img
#     captions = item["captions"]

# # print(captions[0])
    
# base64_image = img2base64(image_url)

# completion = client.chat.completions.create(
#     model="qwen-vl-plus",  # 此处以qwen-vl-plus为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
#     messages=[{"role": "user","content": [
#             {"type": "image_url",
#              "image_url": {"url":f"data:image/jpeg;base64,{base64_image}"}},
#             {"type": "text", "text": "基于原图以及给出的修改caption，生成一句修改后的图片的描述，要求突出修改内容且保留原图中未被修改的核心特征，使用客观的描述语句，不要使用”保留了原图中”这样的语句，而是直接描述，只要图中的dress部分，其他不用描述，使用英文，修改caption如下：" + captions[0]},
#             ]}]
#     )
# output = json.loads(completion.model_dump_json())
# print(output["choices"][0]["message"]["content"])
# # print(completion.model_dump_json()["choices"])