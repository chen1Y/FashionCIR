import json

# 读取原始JSON文件
with open("modified_val_test.json", "r") as f:
    data = json.load(f)

# 构建目标格式的字典
target_dict = {}
for item in data:
    candidate_key = item["candidate"]
    target_key = item["target"]
    final_key = candidate_key + "_" + target_key
    # 这里假设从modified_texts中选取某一条文本，若需其他逻辑可调整
    text = item["modified_texts"] 
    target_dict[final_key] = text

# 将结果写入新JSON文件
with open("converted.json", "w") as f:
    json.dump(target_dict, f, indent=2)
