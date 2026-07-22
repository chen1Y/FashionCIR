import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
import os
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, AutoModelForImageTextToText
from peft import LoraConfig, get_peft_model
import time


class DQU_CIR(nn.Module):
    def __init__(self, hidden_dim=1024, dropout = 0.5, num_heads=8):
        super().__init__()
        # load CLIP backbone for image/text feature extraction
        self.clip, _, _ = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2B-s32B-b79K')
        # print("CLIP loaded with ViT-H-14 backbone",self.clip)
        self.clip = self.clip.float()
        self.tokenizer = open_clip.get_tokenizer('ViT-H-14')
        for param in self.clip.parameters():
            param.requires_grad = False  # 冻结CLIP参数
        # optionally load Qwen3-VL model and freeze most layers
        try:
            # use an appropriate pretrained identifier
            # download or cache model files in current working directory
            cwd = os.getcwd()
            self.qwen = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3-VL-8B-Instruct", 
                                                                        torch_dtype=torch.bfloat16,
                                                                        cache_dir=cwd, 
                                                                        attn_implementation="flash_attention_2", 
                                                                        )
            self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
            # freeze all parameters initially
            lora_config = LoraConfig(
                r=16,  # rank
                lora_alpha=32,
                init_lora_weights="gaussian",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 只微调cross-attention的投影层
                lora_dropout=0.1,
                bias="none"
            )
            self.qwen = get_peft_model(self.qwen, lora_config)
            self.qwen.print_trainable_parameters()  # 打印可训练参数数量
            
            # 添加可学习查询，匹配Qwen3-VL的隐藏层维度 (假设维度为 3584 或 4096 根据具体模型)
            self.qwen_hidden_dim = self.qwen.config.hidden_size  # 从加载的模型配置读取隐藏维度
            self.num_learnable_queries = 256 # 示例数量
            self.learnable_queries = nn.Parameter(torch.randn(1, self.num_learnable_queries, 4096))
            # print("learnable_queries shape: ", self.learnable_queries.shape)
        except ImportError:
            # transformers not installed; skip Qwen3 loading
            self.qwen = None

        # cross-attention module, text queries attend to image features
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

        self.loss_weight = torch.nn.Parameter(torch.FloatTensor((10.,)))

        self.combiner_fc = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim),
                                         nn.ReLU())
        
        # MLLM输出到CLIP维度的投影网络
        if self.qwen is not None:
            self.mllm_proj = nn.Sequential(
                nn.Linear(self.qwen_hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim) 
            )

        self.dropout = nn.Dropout(dropout)
        self.scaler_fc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                       nn.ReLU(),
                                       nn.Dropout(dropout),
                                       nn.Linear(hidden_dim, 1),
                                       nn.Sigmoid())

    # CLIP捕捉图像特征
    def extract_img_fea(self, x):
        image_features = self.clip.encode_image(x)
        # print("image_features shape: ", image_features.shape)
        return image_features

    # CLIP捕捉文本特征
    def extract_text_fea(self, txt, visual_query=None):
        # encode raw text tokens
        txt = self.tokenizer(txt).cuda()
        text_features = self.clip.encode_text(txt)      # [B, D]
        return text_features

    # Hidden-state branch used only for differentiable training.
    def _extract_hidden_text_features(self, textual_query, visual_query_raw):
        # # print("textual_query: ", textual_query)
        # print("visual_query: ", visual_query)
        # print("visual_query shape: ", visual_query.shape)
        """
        textual_query: 使用CLIP提取的文本或者纯文本形式文本 
        visual_query: 使用CLIP预处理后的图像输入
        raw_images: 用于Qwen的原始图片输入 (batch_size, 3, H, W) 或 PIL Image列表
        raw_texts: 用于Qwen的原始文本列表 (str)
        """
        # 使用 Qwen3-VL 融合特征 
        self.processor.tokenizer.padding_side = 'left' 
        batch_size = len(textual_query)
        batch_messages = []
        for i in range(batch_size):
            # print("textual_query[i]: ", textual_query[i])
            # 文本输入如今正确，texual_query如今为修改文本
            # print("Original Text: ", original_text[i])
            # 定义引导思维链的模板
            cot_prompt_wrapper = """你是一个视觉推理专家。请针对输入的参考图像和修改指令，按以下步骤思考并输出：
                                0.图片为时尚领域图片，因此忽略背景，只关注图中的主体。
                                1. 观察参考图中的主体、颜色及布局。
                                2. 分析修改指令，确定要增加、删除或替换的视觉元素，分析修改指令与参考图中矛盾的地方。
                                3. 综合以上两步，写出一句描述目标图像最终形态的完整文本，文本需要简短，只需要描述最终形态就行，例如"a women in a black dress"。
                                修改指令是：{original_instruction}
                                """

            # 在循环中组装
            textual_query[i] = cot_prompt_wrapper.format(original_instruction=textual_query[i])
            # print("Modified textual_query[i]: ", textual_query[i])
            batch_messages.append(
                [{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": visual_query_raw[i], 
                        },
                        {"type": "text", "text": textual_query[i]},
                    ],
                }]
            )
        # print("batch_messages: ", batch_messages)
        inputs = self.processor.apply_chat_template(
            batch_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            padding=True,
        ).to(self.qwen.device)
        # print("inputs: ", inputs)
        # torch.cuda.synchronize() 
        # start_time = time.time()
        
        # Use hidden states directly: generation and decoding are discrete, so
        # they cannot pass the retrieval gradient to Qwen LoRA parameters.
        mllm_outputs = self.qwen(**inputs, output_hidden_states=True, return_dict=True)
        hidden_states = mllm_outputs.hidden_states[-1]
        attention_mask = inputs.attention_mask.to(hidden_states.device)
        last_token = attention_mask.sum(dim=1) - 1
        pooled = hidden_states[
            torch.arange(hidden_states.shape[0], device=hidden_states.device), last_token
        ]
        query = self.mllm_proj(pooled.float())
        return F.normalize(query, p=2, dim=-1)

    def generate_target_text(self, textual_query, visual_query_raw):
        """Generate target-image descriptions for the CLIP inference branch."""
        self.processor.tokenizer.padding_side = 'left'
        batch_messages = []
        for instruction, image in zip(textual_query, visual_query_raw):
            prompt = (
                "You are a fashion retrieval expert. Inspect the reference image and "
                "apply the edit instruction. Output one short English description of "
                "the target garment only. Edit instruction: " + str(instruction)
            )
            batch_messages.append([{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }])

        inputs = self.processor.apply_chat_template(
            batch_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            padding=True,
        ).to(self.qwen.device)
        with torch.no_grad():
            generated_ids = self.qwen.generate(**inputs, max_new_tokens=64)
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    def extract_query(self, textual_query, visual_query):
        """Inference query: Qwen-generated text encoded by CLIP and fused with image."""
        generated_texts = self.generate_target_text(textual_query, visual_query)
        text_features = F.normalize(self.extract_text_fea(generated_texts), p=2, dim=-1)
        image_features = F.normalize(self.extract_img_fea(visual_query), p=2, dim=-1)
        combined_feature = self.combiner_fc(torch.cat([text_features, image_features], dim=-1))
        dynamic_scaler = self.scaler_fc(self.dropout(combined_feature))
        query = dynamic_scaler * text_features + (1 - dynamic_scaler) * image_features
        return F.normalize(query, p=2, dim=-1)

    def extract_hidden_query_fusion(self, textual_query, visual_query, visual_query_raw):
        """Training query: differentiable Qwen hidden states drive LoRA updates."""
        text_features = self._extract_hidden_text_features(textual_query, visual_query_raw)
        image_features = F.normalize(self.extract_img_fea(visual_query), p=2, dim=-1)
        combined_feature = self.combiner_fc(torch.cat([text_features, image_features], dim=-1))
        dynamic_scaler = self.scaler_fc(self.dropout(combined_feature))
        query = dynamic_scaler * text_features + (1 - dynamic_scaler) * image_features
        return F.normalize(query, p=2, dim=-1)

    # 混合特征
    def extract_query_fusion(self, textual_query, visual_query, original_text, visual_query_raw):
        # 两种textual_query
        # 1.原版texual_query
        # textual_query = F.normalize(self.extract_text_fea(original_text), p=2, dim=-1)
        # 2.MLLM生成的textual_query
        textual_query = self.extract_query(textual_query, visual_query_raw)  # 使用MLLM生成的文本特征
        visual_query = F.normalize(self.extract_img_fea(visual_query), p=2, dim=-1)
        combined_feature = self.combiner_fc(torch.cat([textual_query, visual_query], dim=-1))
        dynamic_scaler = self.scaler_fc(self.dropout(combined_feature))
        query = dynamic_scaler * textual_query + (1 - dynamic_scaler) * visual_query
        return F.normalize(query, p=2, dim=-1)

    # 捕捉目标图像特征
    def extract_target(self, target_img):
        target_img_fea = self.extract_img_fea(target_img)
        # print("target_img_fea shape: ", target_img_fea.shape)
        return F.normalize(target_img_fea, p=2, dim=-1)

    # 计算损失
    def compute_loss(self, textual_query, visual_query, target_img, original_text, visual_query_raw):

        # Train Qwen/LoRA through continuous hidden states.  The generated-text
        # route is deliberately reserved for inference because decoded token IDs
        # cannot carry retrieval gradients.
        query_feature = self.extract_hidden_query_fusion(
            textual_query, visual_query, visual_query_raw
        )
        target_feature = self.extract_target(target_img)

        loss = {}
        # loss['ranking'] = self.simple_cosine_loss(query_feature, target_feature)
        # 使用交叉熵损失，考虑到负样本的存在 
        loss['ranking'] = self.ranking_nce_loss(query_feature, target_feature)                                                                                        
        return loss

    def ranking_nce_loss(self, query, target):
        # print("query:", query)
        x = torch.mm(query, target.t())
        # print("x:",x)
        labels = torch.tensor(range(x.shape[0])).long()
        labels = torch.autograd.Variable(labels).cuda()
        loss = F.cross_entropy(self.loss_weight * x, labels)
        return loss

    def simple_cosine_loss(self, query, target):
        # 按元素相乘然后按特征维度求和，得到每个样本对应的相似度
        # query: [B, D], target: [B, D]
        # cosine_sim: [B]
        cosine_sim = torch.sum(query * target, dim=1)
        
        # 损失定义为 1 减去相似度，相似度越高（接近1），损失越低（接近0）
        # 取批次的平均值作为最终的 loss
        loss = torch.mean(1.0 - cosine_sim)
        return loss
    

