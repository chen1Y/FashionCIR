from diffusers import DiffusionPipeline
import torch
import json

print(torch.__version__)
print(torch.cuda.is_available())
device = torch.device("cuda")
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.float16, use_safetensors=True, variant="fp16",low_cpu_mem_usage=True, cache_dir='./')
pipe.enable_xformers_memory_efficient_attention()
pipe.to(device)

caption_file = "../data/FashionIQ/captions/image_captions_dress_train.json"
# modified_file = "../data/FashionIQ/captions/cap.dress.train.json"
modified_file = "../data/FashionIQ/captions/modified_captions_train.json"
output_file = "../data/FashionIQ/modified_image_train_test/"
with open(modified_file,'r',encoding='utf-8') as f:
    modified_data = json.load(f)
with open(caption_file, 'r', encoding='utf-8') as f:
    img_captions = json.load(f)
# captions = list(all_data.values())
i = 0
for item in modified_data:
    # print(item)
    candidate = item['candidate']
    target = item['target']
    candidate_texts = img_captions[candidate]
    # captions = item['captions'][0] + ' and ' + item['captions'][1]
    captions = item['modified_texts']
    # modified_texts = candidate_texts + ' but ' + captions
    modified_texts = f"({candidate_texts}:0.3) BREAK ({captions}:1.5)"
    print(modified_texts)
    # print(all_data[item])
    # prompt = all_data[item]
    img_name = candidate + '_' + target
    save_path = output_file + img_name + ".jpg"
    images = pipe(prompt=modified_texts).images[0]
    images.save(save_path)
    i = i + 1
    if i > 10:
        break
# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()

# prompt = "A black dress with floral patterns features a halter neckline and a draped detail, short-length."

# images = pipe(prompt=prompt).images[0]

# images.save("dress1.png")