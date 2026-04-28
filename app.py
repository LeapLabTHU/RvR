import gradio as gr
import numpy as np
import os
import torch
import random

from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
from PIL import Image

from data.data_utils import add_special_tokens, pil_img2rgb
from data.transforms import ImageTransform
from inferencer import InterleaveInferencer
from modeling.autoencoder import load_ae
from modeling.bagel.qwen2_navit import NaiveCache
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM,
    SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer

import argparse
from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model
from safetensors.torch import load_file as load_safetensors

os.environ["GRADIO_ANALYTICS_ENABLED"] = "false"
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""


RVR_MODEL_PATH = "./models/RvR-7B-MoT"
# BAGEL and RvR share the same model directory; only the EMA weight file differs.
EMA_FILES = {"BAGEL": "ema.safetensors", "RvR": "rvr.safetensors"}

parser = argparse.ArgumentParser() 
parser.add_argument("--server_name", type=str, default="0.0.0.0")
parser.add_argument("--server_port", type=int, default=443)
parser.add_argument("--share", action="store_true")
parser.add_argument("--model_path", type=str, default=RVR_MODEL_PATH)
parser.add_argument("--mode", type=int, default=1)
parser.add_argument("--zh", action="store_true")
args = parser.parse_args()

model_path = args.model_path

llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
llm_config.qk_norm = True
llm_config.tie_word_embeddings = False
llm_config.layer_module = "Qwen2MoTDecoderLayer"

vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
vit_config.rope = False
vit_config.num_hidden_layers -= 1

vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

config = BagelConfig(
    visual_gen=True,
    visual_und=True,
    llm_config=llm_config, 
    vit_config=vit_config,
    vae_config=vae_config,
    vit_max_num_patch_per_side=70,
    connector_act='gelu_pytorch_tanh',
    latent_patch_size=2,
    max_latent_size=64,
)

with init_empty_weights():
    language_model = Qwen2ForCausalLM(llm_config)
    vit_model      = SiglipVisionModel(vit_config)
    model          = Bagel(language_model, vit_model, config)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

vae_transform = ImageTransform(1024, 512, 16)
vit_transform = ImageTransform(980, 224, 14)

# Model loading and multi-GPU inference setup
device_map = infer_auto_device_map(
    model,
    max_memory={i: "80GiB" for i in range(torch.cuda.device_count())},
    no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
)

same_device_modules = [
    'language_model.model.embed_tokens',
    'time_embedder',
    'latent_pos_embed',
    'vae2llm',
    'llm2vae',
    'connector',
    'vit_pos_embed'
]

if torch.cuda.device_count() == 1:
    first_device = device_map.get(same_device_modules[0], "cuda:0")
    for k in same_device_modules:
        if k in device_map:
            device_map[k] = first_device
        else:
            device_map[k] = "cuda:0"
else:
    first_device = device_map.get(same_device_modules[0])
    for k in same_device_modules:
        if k in device_map:
            device_map[k] = first_device

# Initially load BAGEL's EMA weights (used for T2I); rvr.safetensors will be hot-swapped in when switching to RvR.
_initial_ema = os.path.join(model_path, EMA_FILES["BAGEL"])

if args.mode == 1:
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=_initial_ema,
        device_map=device_map,
        offload_buffers=True,
        offload_folder="offload",
        dtype=torch.bfloat16,
        force_hooks=True,
    ).eval()
elif args.mode == 2:  # NF4 quantization
    bnb_quantization_config = BnbQuantizationConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=False, bnb_4bit_quant_type="nf4")
    model = load_and_quantize_model(
        model, 
        weights_location=_initial_ema,
        bnb_quantization_config=bnb_quantization_config,
        device_map=device_map,
        offload_folder="offload",
    ).eval()
elif args.mode == 3:  # INT8 quantization
    bnb_quantization_config = BnbQuantizationConfig(load_in_8bit=True, torch_dtype=torch.float32)
    model = load_and_quantize_model(
        model, 
        weights_location=_initial_ema,
        bnb_quantization_config=bnb_quantization_config,
        device_map=device_map,
        offload_folder="offload",
    ).eval()
else:
    raise NotImplementedError

current_model_name = "BAGEL"


def reload_model_ema(model_choice):
    global current_model_name
    if model_choice == current_model_name:
        return f"✅ Current model is: {model_choice}"

    ema_path = os.path.join(model_path, EMA_FILES[model_choice])
    if not os.path.exists(ema_path):
        return f"❌ File not found: {ema_path}"

    print(f"[reload] Loading EMA from: {ema_path}")
    state_dict = load_safetensors(ema_path, device="cpu")

    for name, param in model.named_parameters():
        if name in state_dict:
            param.data.copy_(state_dict[name].to(device=param.device, dtype=param.dtype))
    for name, buf in model.named_buffers():
        if name in state_dict:
            buf.data.copy_(state_dict[name].to(device=buf.device, dtype=buf.dtype))

    del state_dict
    torch.cuda.empty_cache()

    current_model_name = model_choice
    print(f"[reload] Model switched to: {model_choice}")
    return f"✅ Successfully switched to: {model_choice}"


# Inferencer setup
inferencer = InterleaveInferencer(
    model=model,
    vae_model=vae_model,
    tokenizer=tokenizer,
    vae_transform=vae_transform,
    vit_transform=vit_transform,
    new_token_ids=new_token_ids,
)


def set_seed(seed):
    """Set random seeds for reproducibility."""
    if seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


REFINE_PROMPT = '''Here is the user's prompt: {user_prompt}. '''


def text_to_image(prompt, cfg_text_scale=4.0, cfg_interval=0.4,
                 timestep_shift=3.0, num_timesteps=50,
                 cfg_renorm_min=0.0, cfg_renorm_type="global",
                 seed=0, image_ratio="1:1"):
    set_seed(seed)

    if image_ratio == "1:1":
        image_shapes = (1024, 1024)
    elif image_ratio == "4:3":
        image_shapes = (768, 1024)
    elif image_ratio == "3:4":
        image_shapes = (1024, 768)
    elif image_ratio == "16:9":
        image_shapes = (576, 1024)
    elif image_ratio == "9:16":
        image_shapes = (1024, 576)

    inference_hyper = dict(
        max_think_token_n=1024,
        do_sample=False,
        text_temperature=0.3,
        cfg_text_scale=cfg_text_scale,
        cfg_interval=[cfg_interval, 1.0],  # interval end is fixed at 1.0
        timestep_shift=timestep_shift,
        num_timesteps=num_timesteps,
        cfg_renorm_min=cfg_renorm_min,
        cfg_renorm_type=cfg_renorm_type,
        image_shapes=image_shapes,
    )

    result = inferencer(text=prompt, think=False, **inference_hyper)
    return result["image"]


# Helper for loading example images.
def load_example_image(image_path):
    try:
        return Image.open(image_path)
    except Exception as e:
        print(f"Error loading example image: {e}")
        return None


def image_refinement(image, prompt, cfg_text_scale=4.0, cfg_img_scale=2.0,
                     cfg_interval=0.0, cfg_renorm_min=0.0, cfg_renorm_type="text_channel",
                     timestep_shift=3.0, num_timesteps=50, seed=43):
    global current_model_name

    if current_model_name != "RvR":
        status = reload_model_ema("RvR")
        if "❌" in status:
            return None, status

    set_seed(seed)

    if image is None:
        return None, "❌ Please provide an initial image first."

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    image = pil_img2rgb(image)

    formatted_prompt = REFINE_PROMPT.format(user_prompt=prompt)

    inference_hyper = dict(
        max_think_token_n=1024,
        do_sample=False,
        text_temperature=0.3,
        cfg_text_scale=cfg_text_scale,
        cfg_img_scale=cfg_img_scale,
        cfg_interval=[cfg_interval, 1.0],
        timestep_shift=timestep_shift,
        num_timesteps=num_timesteps,
        cfg_renorm_min=cfg_renorm_min,
        cfg_renorm_type=cfg_renorm_type,
    )

    result = inferencer(
        image=image,
        text=formatted_prompt,
        think=False,
        align_output=True,
        wo_vae=True,
        **inference_hyper,
    )
    return result["image"], "✅ Refinement complete"


# Gradio UI
import base64

def _load_banner_b64(path="assets/banner.png"):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

_banner_b64 = _load_banner_b64()

with gr.Blocks() as demo:
    gr.Markdown(f"""
<div>
  <img src="data:image/png;base64,{_banner_b64}" alt="RvR: Refinement via Regeneration" width="640"/>
</div>
""")

    txt_input = gr.Textbox(
        label="Prompt",
        placeholder="Enter your prompt here...",
    )

    with gr.Row():
        # ── Left column: T2I ──
        with gr.Column(scale=1):
            gr.Markdown("### Initial Generation (BAGEL) or Upload An Image")
            img_output = gr.Image(label="Initial Image", interactive=True, height=400)
            gen_btn = gr.Button("Generate", variant="primary")

            with gr.Accordion("T2I Hyperparameters", open=False):
                with gr.Group():
                    with gr.Row():
                        seed = gr.Slider(minimum=0, maximum=1000000, value=0, step=1,
                                         label="Seed", info="0 for random seed, positive for reproducible results")
                        image_ratio = gr.Dropdown(choices=["1:1", "4:3", "3:4", "16:9", "9:16"],
                                                  value="1:1", label="Image Ratio",
                                                  info="The longer size is fixed to 1024")
                    with gr.Row():
                        cfg_text_scale = gr.Slider(minimum=1.0, maximum=8.0, value=4.0, step=0.1, interactive=True,
                                                   label="CFG Text Scale", info="Controls how strongly the model follows the text prompt (4.0-8.0)")
                        cfg_interval = gr.Slider(minimum=0.0, maximum=1.0, value=0.4, step=0.1,
                                                 label="CFG Interval", info="Start of CFG application interval (end is fixed at 1.0)")
                    with gr.Row():
                        cfg_renorm_type = gr.Dropdown(choices=["global", "local", "text_channel"],
                                                      value="global", label="CFG Renorm Type",
                                                      info="If the genrated image is blurry, use 'global'")
                        cfg_renorm_min = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.1, interactive=True,
                                                   label="CFG Renorm Min", info="1.0 disables CFG-Renorm")
                    with gr.Row():
                        num_timesteps = gr.Slider(minimum=10, maximum=100, value=50, step=5, interactive=True,
                                                  label="Timesteps", info="Total denoising steps")
                        timestep_shift = gr.Slider(minimum=1.0, maximum=5.0, value=3.0, step=0.5, interactive=True,
                                                   label="Timestep Shift", info="Higher values for layout, lower for details")

        # ── Right column: Refine ──
        with gr.Column(scale=1):
            gr.Markdown("### Refinement (RvR)")
            refined_img_output = gr.Image(label="Refined Image", height=400)
            refine_btn = gr.Button("Refine", variant="primary")

            with gr.Accordion("Refine Hyperparameters", open=False):
                with gr.Group():
                    with gr.Row():
                        refine_seed = gr.Slider(minimum=0, maximum=1000000, value=0, step=1,
                                                label="Seed", info="0 for random seed, positive for reproducible results")
                        refine_cfg_text_scale = gr.Slider(minimum=1.0, maximum=8.0, value=4.0, step=0.1, interactive=True,
                                                          label="CFG Text Scale", info="Controls how strongly the model follows the text prompt")
                    with gr.Row():
                        refine_cfg_img_scale = gr.Slider(minimum=1.0, maximum=4.0, value=2.0, step=0.1, interactive=True,
                                                          label="CFG Image Scale", info="Controls how much the model preserves input image details")
                        refine_cfg_interval = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.1, interactive=True,
                                                         label="CFG Interval", info="Start of CFG application interval (end is fixed at 1.0)")
                    with gr.Row():
                        refine_cfg_renorm_type = gr.Dropdown(choices=["global", "local", "text_channel"],
                                                              value="text_channel", label="CFG Renorm Type",
                                                              info="If the genrated image is blurry, use 'global'")
                        refine_cfg_renorm_min = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.1, interactive=True,
                                                           label="CFG Renorm Min", info="1.0 disables CFG-Renorm")
                    with gr.Row():
                        refine_num_timesteps = gr.Slider(minimum=10, maximum=100, value=50, step=5, interactive=True,
                                                          label="Timesteps", info="Total denoising steps")
                        refine_timestep_shift = gr.Slider(minimum=1.0, maximum=10.0, value=3.0, step=0.5, interactive=True,
                                                           label="Timestep Shift", info="Higher values for layout, lower for details")

    # ── Event handlers ──
    def process_text_to_image(prompt, cfg_text_scale, cfg_interval, timestep_shift,
                              num_timesteps, cfg_renorm_min, cfg_renorm_type, seed, image_ratio):
        if current_model_name != "BAGEL":
            reload_model_ema("BAGEL")
        return text_to_image(
            prompt, cfg_text_scale, cfg_interval,
            timestep_shift, num_timesteps,
            cfg_renorm_min, cfg_renorm_type,
            seed, image_ratio,
        )

    gr.on(
        triggers=[gen_btn.click, txt_input.submit],
        fn=process_text_to_image,
        inputs=[
            txt_input, cfg_text_scale, cfg_interval, timestep_shift,
            num_timesteps, cfg_renorm_min, cfg_renorm_type, seed, image_ratio,
        ],
        outputs=[img_output],
    )

    def process_refine(image, prompt, cfg_text_scale, cfg_img_scale, cfg_interval,
                       cfg_renorm_min, cfg_renorm_type, timestep_shift, num_timesteps, seed):
        refined_image, _ = image_refinement(
            image, prompt,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_interval=cfg_interval,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            timestep_shift=timestep_shift,
            num_timesteps=num_timesteps,
            seed=seed,
        )
        return refined_image

    refine_btn.click(
        fn=process_refine,
        inputs=[
            img_output, txt_input,
            refine_cfg_text_scale, refine_cfg_img_scale, refine_cfg_interval,
            refine_cfg_renorm_min, refine_cfg_renorm_type,
            refine_timestep_shift, refine_num_timesteps, refine_seed,
        ],
        outputs=[refined_img_output],
    )

    gr.Examples(
        examples=[
            [
                'Please create a sculpture. The main body is a robot that imitates Rodin\'s "The Thinker", but it has no melancholy expression. The whole body is made of transparent glass and has complex golden gears running inside, in a steampunk style.',
                "assets/examples/case1.png",
            ],
            [
                "The huge elephant has no wings, but its body is filled with shining stars and slowly floats towards the night sky.",
                "assets/examples/case2.png",
            ],
        ],
        inputs=[txt_input, img_output],
        label="Examples",
    )

    gr.Markdown("""
<div style="display: flex; justify-content: flex-start; flex-wrap: wrap; gap: 10px;">
  <a href="https://arxiv.org/abs/2505.14683">
    <img
      src="https://img.shields.io/badge/RvR-Paper-red?logo=arxiv&logoColor=red"
      alt="RvR Paper on arXiv"
    />
  </a>
  <a href="https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT">
    <img
        src="https://img.shields.io/badge/RvR-Hugging%20Face-orange?logo=huggingface&logoColor=yellow"
        alt="RvR on Hugging Face"
    />
  </a>
</div>
""")

UI_TRANSLATIONS = {
    "Prompt":"提示词",
    "T2I Hyperparameters":"T2I 推理参数",
    "Seed":"随机种子",
    "0 for random seed, positive for reproducible results":"0为随机种子，正数表示可重复结果",
    "Image Ratio":"图片比例",
    "The longer size is fixed to 1024":"长边固定为1024",
    "CFG Text Scale":"文本CFG强度",
    "Controls how strongly the model follows the text prompt (4.0-8.0)":"控制模型是否遵循文本提示（4.0-8.0）",
    "CFG Interval":"CFG应用间隔",
    "Start of CFG application interval (end is fixed at 1.0)":"CFG应用间隔的开始（结束固定为1.0）",
    "CFG Renorm Type":"CFG 重归一化类型",
    "If the genrated image is blurry, use 'global'":"如果生成的图像模糊，请使用'global'",
    "CFG Renorm Min":"CFG 重归一化最小值",
    "1.0 disables CFG-Renorm":"1.0 禁用 CFG 重归一化",
    "Timesteps":"时间步数",
    "Total denoising steps":"总去噪步数",
    "Timestep Shift":"时间步偏移",
    "Higher values for layout, lower for details":"值更大更倾向于调整布局，值更小更倾向于调整细节",
    "Generate":"开始生成",
    "Initial Image":"初始图像",
    "Refine Hyperparameters":"重生成参数",
    "Controls how strongly the model follows the text prompt":"控制模型是否遵循文本提示的强度",
    "CFG Image Scale":"图像CFG强度",
    "Controls how much the model preserves input image details":"控制模型保留输入图像细节的强度",
    "Refine":"开始重生成",
    "Refined Image":"重生成图像",
}

def apply_localization(block):
    def process_component(component):
        if not component:
            return
        for attr in ['label', 'info', 'placeholder']:
            if hasattr(component, attr):
                text = getattr(component, attr)
                if text in UI_TRANSLATIONS:
                    setattr(component, attr, UI_TRANSLATIONS[text])
        if hasattr(component, 'children'):
            for child in component.children:
                process_component(child)
    process_component(block)
    return block

if __name__ == "__main__":
    if args.zh:
        demo = apply_localization(demo)
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        inbrowser=False,
    )
