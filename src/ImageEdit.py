import torch
from diffusers import Step1XEditPipelineV1P2
from diffusers.utils import load_image
pipe = Step1XEditPipelineV1P2.from_pretrained("stepfun-ai/Step1X-Edit-v1p2", torch_dtype=torch.bfloat16)
pipe.to("cuda")
print("=== processing image ===")
image = load_image("examples/0000.jpg").convert("RGB")
prompt = "add a ruby pendant on the girl's neck."
enable_thinking_mode=True
enable_reflection_mode=True
pipe_output = pipe(
    image=image,
    prompt=prompt,
    num_inference_steps=50,
    true_cfg_scale=6,
    generator=torch.Generator().manual_seed(42),
    enable_thinking_mode=enable_thinking_mode,
    enable_reflection_mode=enable_reflection_mode,
)
if enable_thinking_mode:
    print("Reformat Prompt:", pipe_output.reformat_prompt)
for image_idx in range(len(pipe_output.images)):
    pipe_output.images[image_idx].save(f"0001-{image_idx}.jpg", lossless=True)
    if enable_reflection_mode:
        print(pipe_output.think_info[image_idx])
        print(pipe_output.best_info[image_idx])
pipe_output.final_images[0].save(f"0001-final.jpg", lossless=True)