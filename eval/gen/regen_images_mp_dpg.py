import os
import json
import argparse
from safetensors.torch import load_file

import torch
import torch.distributed as dist
from data.data_utils import add_special_tokens
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.autoencoder import load_ae
from inferencer import InterleaveInferencer

from PIL import Image
from tqdm import tqdm
from data.transforms import ImageTransform


def move_generation_input_to_device(generation_input, device):
    for k, v in generation_input.items():
        if isinstance(v, torch.Tensor):
            generation_input[k] = v.to(device)
    return generation_input


def setup_distributed():
    from datetime import timedelta
    timeout = timedelta(seconds=int(os.environ.get("TORCH_DISTRIBUTED_TIMEOUT", 3600)))
    dist.init_process_group(backend="nccl", timeout=timeout)
    # Fall back to RANK % ngpus when LOCAL_RANK is not set.
    local_rank_env = os.environ.get("LOCAL_RANK")
    if local_rank_env is None:
        rank = int(os.environ.get("RANK", 0))
        num_gpus = torch.cuda.device_count() or 1
        torch.cuda.set_device(rank % num_gpus)
    else:
        torch.cuda.set_device(int(local_rank_env))


REFINE_PROMPT = '''Here is the user's prompt: {user_prompt}. '''

def generate_image_from_condition(prompt, cond_image: Image.Image, *, cfg_text_scale=4.0, cfg_img_scale=2.0, cfg_interval=[0, 1.0], cfg_renorm_min=0., cfg_renorm_type="text_channel", timestep_shift=3.0, num_timesteps=50, resolution=1024):
    formatted_prompt = REFINE_PROMPT.format(user_prompt=prompt)
    inference_hyper = dict(
        max_think_token_n=1024,
        do_sample=False,
        text_temperature=0.3,
        cfg_text_scale=cfg_text_scale,
        cfg_img_scale=cfg_img_scale,
        cfg_interval=cfg_interval,
        timestep_shift=timestep_shift,
        num_timesteps=num_timesteps,
        cfg_renorm_min=cfg_renorm_min,
        cfg_renorm_type=cfg_renorm_type,
    )
    result = inferencer(
        image=cond_image,
        text=formatted_prompt,
        think=False,
        align_output=True,
        wo_vae=True,
        **inference_hyper,
    )
    return result["image"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerate DPG images conditioned on previous outputs using Bagel model.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save regenerated images.")
    parser.add_argument("--prompts_file", type=str, required=True, help="JSON mapping prompt key (file basename) to prompt text.")
    parser.add_argument("--input_base_dir", type=str, required=True, help="Input image base directory.")
    parser.add_argument("--num_images", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--cfg_text_scale", type=float, default=4.0)
    parser.add_argument("--cfg_img_scale", type=float, default=2.0)
    parser.add_argument("--cfg_interval", type=float, default=0.0)
    parser.add_argument("--cfg_renorm_min", type=float, default=0.0)
    parser.add_argument("--timestep_shift", type=float, default=3.0)
    parser.add_argument("--num_timesteps", type=int, default=50)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--max_latent_size", type=int, default=64)
    parser.add_argument('--model-path', type=str, default='./models/RvR-7B-MoT')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg_renorm_type", type=str, default="text_channel")

    args = parser.parse_args()

    seed = args.seed
    if seed is not None:
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f"cuda:{rank}"

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    if rank == 0:
        print(f"Output images are saved in {output_dir}")

    llm_config = Qwen2Config.from_json_file(os.path.join(args.model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(args.model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    vae_model, vae_config = load_ae(local_path=os.path.join(args.model_path, "ae.safetensors"))

    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
        latent_patch_size=2,
        max_latent_size=args.max_latent_size,
    )
    language_model = Qwen2ForCausalLM(llm_config)
    vit_model = SiglipVisionModel(vit_config)
    model = Bagel(language_model, vit_model, config)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

    tokenizer = Qwen2Tokenizer.from_pretrained(args.model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    model_state_dict_path = os.path.join(args.model_path, "rvr.safetensors")
    model_state_dict = load_file(model_state_dict_path, device="cpu")
    msg = model.load_state_dict(model_state_dict, strict=False)
    if rank == 0:
        print(msg)
    del model_state_dict

    amp_dtype = torch.bfloat16
    model = model.to(dtype=amp_dtype).to(device).eval()
    vae_model = vae_model.to(dtype=amp_dtype).to(device).eval()
    gen_model = model

    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 224, 14)
    inferencer = InterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )

    # Locally patch inferencer methods to (1) move prepare_* tensors to this rank's
    # device and (2) wrap forwards in bf16 autocast for safety.
    import types as _types
    _device_obj = torch.device(device)

    def _move_to_dev(d):
        return move_generation_input_to_device(d, _device_obj)

    def _update_context_text_patched(self, text, gen_context):
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        generation_input, kv_lens, ropes = self.model.prepare_prompts(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            prompts=[text],
            tokenizer=self.tokenizer,
            new_token_ids=self.new_token_ids,
        )
        generation_input = _move_to_dev(generation_input)
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=True, dtype=amp_dtype):
                past_key_values = self.model.forward_cache_update_text(past_key_values, **generation_input)
        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
        return gen_context

    def _update_context_image_patched(self, image, gen_context, vae=True, vit=True):
        assert vae or vit
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        if vae:
            print(f"Preparing vae images for {image}")
            generation_input, kv_lens, ropes = self.model.prepare_vae_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes,
                images=[image],
                transforms=self.vae_transform,
                new_token_ids=self.new_token_ids,
            )
            generation_input = _move_to_dev(generation_input)
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=True, dtype=amp_dtype):
                    past_key_values = self.model.forward_cache_update_vae(self.vae_model, past_key_values, **generation_input)
        if vit:
            generation_input, kv_lens, ropes = self.model.prepare_vit_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes,
                images=[image],
                transforms=self.vit_transform,
                new_token_ids=self.new_token_ids,
            )
            generation_input = _move_to_dev(generation_input)
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=True, dtype=amp_dtype):
                    past_key_values = self.model.forward_cache_update_vit(past_key_values, **generation_input)
        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
        return gen_context

    def _gen_image_patched(self, image_shape, gen_context, cfg_text_scale=4.0, cfg_img_scale=1.5, cfg_text_precontext=None, cfg_img_precontext=None, cfg_interval=(0.4, 1.0), cfg_renorm_min=0.0, cfg_renorm_type="global", num_timesteps=50, timestep_shift=3.0):
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        generation_input = self.model.prepare_vae_latent(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            image_sizes=[image_shape],
            new_token_ids=self.new_token_ids,
        )
        generation_input = _move_to_dev(generation_input)

        cfg_text_past_key_values = cfg_text_precontext['past_key_values']
        kv_lens_cfg = cfg_text_precontext['kv_lens']
        ropes_cfg = cfg_text_precontext['ropes']
        generation_input_cfg_text = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg,
            image_sizes=[image_shape],
        )
        generation_input_cfg_text = _move_to_dev(generation_input_cfg_text)

        cfg_img_past_key_values = cfg_img_precontext['past_key_values']
        kv_lens_cfg = cfg_img_precontext['kv_lens']
        ropes_cfg = cfg_img_precontext['ropes']
        generation_input_cfg_img = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg,
            image_sizes=[image_shape],
        )
        generation_input_cfg_img = _move_to_dev(generation_input_cfg_img)

        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=True, dtype=amp_dtype):
                unpacked_latent = self.model.generate_image(
                    past_key_values=past_key_values,
                    cfg_text_past_key_values=cfg_text_past_key_values,
                    cfg_img_past_key_values=cfg_img_past_key_values,
                    num_timesteps=num_timesteps,
                    cfg_text_scale=cfg_text_scale,
                    cfg_img_scale=cfg_img_scale,
                    cfg_interval=cfg_interval,
                    cfg_renorm_min=cfg_renorm_min,
                    cfg_renorm_type=cfg_renorm_type,
                    timestep_shift=timestep_shift,
                    **generation_input,
                    cfg_text_packed_position_ids=generation_input_cfg_text['cfg_packed_position_ids'],
                    cfg_text_packed_query_indexes=generation_input_cfg_text['cfg_packed_query_indexes'],
                    cfg_text_key_values_lens=generation_input_cfg_text['cfg_key_values_lens'],
                    cfg_text_packed_key_value_indexes=generation_input_cfg_text['cfg_packed_key_value_indexes'],
                    cfg_img_packed_position_ids=generation_input_cfg_img['cfg_packed_position_ids'],
                    cfg_img_packed_query_indexes=generation_input_cfg_img['cfg_packed_query_indexes'],
                    cfg_img_key_values_lens=generation_input_cfg_img['cfg_key_values_lens'],
                    cfg_img_packed_key_value_indexes=generation_input_cfg_img['cfg_packed_key_value_indexes'],
                )
                image = self.decode_image(unpacked_latent[0], image_shape)
        return image

    def _gen_text_patched(self, gen_context, max_length: int = 500, do_sample: bool = True, temperature: float = 1.0):
        from copy import deepcopy as _dc
        gen_context = _dc(gen_context)
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        generation_input = self.model.prepare_start_tokens(kv_lens, ropes, self.new_token_ids)
        generation_input = _move_to_dev(generation_input)
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=True, dtype=amp_dtype):
                unpacked_latent = self.model.generate_text(
                    past_key_values=past_key_values,
                    max_length=max_length,
                    do_sample=do_sample,
                    temperature=temperature,
                    end_token_id=self.new_token_ids['eos_token_id'],
                    **generation_input,
                )
        output = self.tokenizer.decode(unpacked_latent[:,0])
        output = output.split('<|im_end|>')[0].split('<|im_start|>')[1]
        return output

    inferencer.update_context_text = _types.MethodType(_update_context_text_patched, inferencer)
    inferencer.update_context_image = _types.MethodType(_update_context_image_patched, inferencer)
    inferencer.gen_image = _types.MethodType(_gen_image_patched, inferencer)
    inferencer.gen_text = _types.MethodType(_gen_text_patched, inferencer)

    cfg_text_scale = args.cfg_text_scale
    cfg_img_scale = args.cfg_img_scale
    cfg_interval = [args.cfg_interval, 1.0]
    timestep_shift = args.timestep_shift
    num_timesteps = args.num_timesteps
    cfg_renorm_min = args.cfg_renorm_min

    try:
        with open(args.prompts_file, 'r') as f:
            dataset = json.load(f)
            if rank == 0:
                print(f"Loaded {len(dataset)} prompts from {args.prompts_file}")
    except Exception as e:
        if rank == 0:
            print(f"Error loading prompts file: {e}")
        dataset = {"default.txt": "a dog on the left and a cat on the right."}

    dataset_items = list(dataset.items())
    total_metadatas = len(dataset_items)

    prompts_per_gpu = (total_metadatas + world_size - 1) // world_size
    start = rank * prompts_per_gpu
    end = min(start + prompts_per_gpu, total_metadatas)
    print(f"GPU {rank}: Processing {end - start} prompts (indices {start} to {end - 1})")

    valid_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
    base_dir = args.input_base_dir
    name_to_images = {}
    try:
        for fname in os.listdir(base_dir):
            fpath = os.path.join(base_dir, fname)
            if not os.path.isfile(fpath):
                continue
            root, ext = os.path.splitext(fname)
            if ext.lower() not in valid_exts:
                continue
            # Expected filename: {key_base}_{idx}.ext; split on the last underscore.
            parts = root.rsplit('_', 1)
            if len(parts) == 2 and parts[1].isdigit():
                key_base = parts[0]
            else:
                key_base = root
            name_to_images.setdefault(key_base, []).append(fpath)
        for k in list(name_to_images.keys()):
            name_to_images[k].sort()
    except Exception as e:
        if rank == 0:
            print(f"Warning: failed to scan input_base_dir: {e}")

    processed_items = dataset_items[start:end]
    for key, prompt in tqdm(processed_items, desc=f"GPU {rank} processing"):
        key_base = key.split('.')[-2]
        print(f"GPU {rank} processing prompt: '{prompt}' (key={key_base})")

        skip_all = True
        for idx_img in range(args.num_images):
            out_filename = f"{key_base}_{idx_img}.jpg"
            out_path = os.path.join(output_dir, out_filename)
            if not os.path.exists(out_path):
                skip_all = False
                break
        if skip_all:
            print(f"GPU {rank} skipping regeneration for prompt: {prompt}")
            continue

        cond_list = name_to_images.get(key_base, [])
        if len(cond_list) == 0:
            print(f"GPU {rank} no condition images found for key={key_base} under {base_dir}, skip")
            continue

        needed = args.num_images
        use_list = []
        while len(use_list) < needed:
            for p in cond_list:
                use_list.append(p)
                if len(use_list) >= needed:
                    break

        # Resume-friendly: skip already-existing outputs, regenerate only missing indices.
        for idx_img in range(args.num_images):
            out_filename = f"{key_base}_{idx_img}.jpg"
            out_path = os.path.join(output_dir, out_filename)
            if os.path.exists(out_path):
                print(f"GPU {rank} skip existing: {out_path}")
                continue
            try:
                cond_image = Image.open(use_list[idx_img]).convert('RGB')
            except Exception as e:
                print(f"GPU {rank} failed to open condition image for index {idx_img}: {e}")
                continue
            gen_image = generate_image_from_condition(
                prompt=prompt,
                cond_image=cond_image,
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=cfg_img_scale,
                cfg_interval=cfg_interval,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=args.cfg_renorm_type,
                timestep_shift=timestep_shift,
                num_timesteps=num_timesteps,
                resolution=args.resolution,
            )
            try:
                gen_image = gen_image.crop(gen_image.getbbox())
            except Exception:
                pass
            gen_image.save(out_path)
            print(f"GPU {rank} saved image to: {out_path}")

    print(f"GPU {rank} has completed all tasks")
    try:
        if dist.is_available() and dist.is_initialized():
            try:
                local_device = torch.cuda.current_device()
            except Exception:
                local_device = 0
            dist.barrier(device_ids=[local_device])
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


