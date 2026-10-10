import argparse
import json
import os
import os.path as osp
import re
import sys
import time

REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import cv2
import numpy as np
import torch
import tqdm
from pytorch_lightning import seed_everything

from conf import HF_HOME, HF_TOKEN
from tools.run_infinity import *

# Set environment variables.
os.environ['HF_TOKEN'] = HF_TOKEN
os.environ['HF_HOME'] = HF_HOME
os.environ['XFORMERS_FORCE_DISABLE_TRITON'] = '1'


def natural_sort_key(value):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r'(\d+)', value)
    ]


def normalize_image_ext(image_ext):
    image_ext = (image_ext or 'png').strip().lower()
    if image_ext.startswith('.'):
        image_ext = image_ext[1:]
    if image_ext not in {'png', 'jpg', 'jpeg'}:
        raise ValueError(f'Unsupported --image_ext={image_ext!r}; choose png, jpg or jpeg')
    return image_ext


def load_dpg_prompts(prompt_dir):
    prompt_dir = osp.abspath(prompt_dir)
    if not osp.isdir(prompt_dir):
        raise FileNotFoundError(f'DPG prompt directory not found: {prompt_dir}')

    prompt_files = [
        osp.join(prompt_dir, filename)
        for filename in os.listdir(prompt_dir)
        if filename.endswith('.txt')
    ]
    prompt_files = sorted(prompt_files, key=lambda path: natural_sort_key(osp.splitext(osp.basename(path))[0]))
    if not prompt_files:
        raise ValueError(f'No .txt prompts found under {prompt_dir}')

    metadatas = []
    for prompt_file in prompt_files:
        prompt_id = osp.splitext(osp.basename(prompt_file))[0]
        with open(prompt_file, 'r', encoding='utf-8') as fp:
            prompt = fp.read().strip()
        if not prompt:
            raise ValueError(f'Empty DPG prompt file: {prompt_file}')
        metadatas.append(
            {
                'id': prompt_id,
                'prompt': prompt,
                'prompt_file': prompt_file,
            }
        )
    return metadatas


def save_metadata_atomic(save_metadata_file_path, save_metadatas):
    tmp_metadata_file_path = save_metadata_file_path + '.tmp'
    with open(tmp_metadata_file_path, 'w', encoding='utf-8') as fp:
        json.dump(save_metadatas, fp)
        fp.write('\n')
    os.replace(tmp_metadata_file_path, save_metadata_file_path)


def image_to_bgr_array(image):
    if isinstance(image, torch.Tensor):
        arr = image.detach().cpu().numpy()
    elif hasattr(image, 'convert'):
        arr = np.array(image.convert('RGB'))
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    else:
        arr = np.asarray(image)

    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[0] < min(arr.shape[1], arr.shape[2]):
        arr = np.transpose(arr, (1, 2, 0))

    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = cv2.cvtColor(arr[:, :, 0], cv2.COLOR_GRAY2BGR)
    elif arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    elif arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f'Unsupported generated image shape: {arr.shape}')

    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if arr.size and arr.max() <= 1.0 and arr.min() >= 0.0:
            arr = arr * 255.0
        elif arr.size and arr.max() <= 1.0 and arr.min() >= -1.0:
            arr = (arr + 1.0) * 127.5
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(arr)


def make_dpg_grid(images, tile_resolution):
    if len(images) != 4:
        raise ValueError(f'DPG-Bench expects exactly 4 images per prompt, got {len(images)}')

    if tile_resolution <= 0:
        tile_resolution = int(images[0].shape[0])

    resized = []
    for image in images:
        if image.shape[0] != tile_resolution or image.shape[1] != tile_resolution:
            interpolation = cv2.INTER_AREA if max(image.shape[:2]) > tile_resolution else cv2.INTER_CUBIC
            image = cv2.resize(image, (tile_resolution, tile_resolution), interpolation=interpolation)
        resized.append(image)

    top = np.concatenate([resized[0], resized[1]], axis=1)
    bottom = np.concatenate([resized[2], resized[3]], axis=1)
    return np.concatenate([top, bottom], axis=0)


def grid_metadata(prompt_metadata, grid_image_path, sample_image_paths, tile_resolution, prompt_for_generation):
    metadata = dict(prompt_metadata)
    metadata['gen_image_paths'] = [grid_image_path]
    metadata['grid_image_path'] = grid_image_path
    metadata['sample_image_paths'] = sample_image_paths
    metadata['n_samples'] = 4
    metadata['tile_resolution'] = tile_resolution
    metadata['prompt_for_generation'] = prompt_for_generation
    return metadata


def complete_grid_path(metadata):
    grid_image_path = metadata.get('grid_image_path')
    if not grid_image_path:
        gen_image_paths = metadata.get('gen_image_paths', [])
        if gen_image_paths:
            grid_image_path = gen_image_paths[0]
    if not grid_image_path:
        return None
    if os.path.isfile(grid_image_path) and os.path.getsize(grid_image_path) > 0:
        return grid_image_path
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    parser.add_argument('--outdir', type=str, default='')
    parser.add_argument('--n_samples', type=int, default=4)
    parser.add_argument('--prompt_dir', type=str, default='evaluation/dpg_bench/prompts')
    parser.add_argument('--metadata_file', type=str, default='')
    parser.add_argument('--image_dir_name', type=str, default='images')
    parser.add_argument('--image_ext', type=str, default='png')
    parser.add_argument('--grid_resolution', type=int, default=0)
    parser.add_argument('--save_individual', type=int, default=0, choices=[0, 1])
    parser.add_argument('--rewrite_prompt', type=int, default=0, choices=[0, 1])
    args = parser.parse_args()

    if args.metadata_file:
        args.prompt_dir = args.metadata_file
    args.image_ext = normalize_image_ext(args.image_ext)
    if args.n_samples != 4:
        raise ValueError('DPG-Bench generation requires --n_samples 4 for a 2x2 evaluation grid')
    if not args.outdir:
        raise ValueError('--outdir is required')

    seed_everything(args.seed)

    # Parse cfg.
    args.cfg = list(map(float, args.cfg.split(',')))
    if len(args.cfg) == 1:
        args.cfg = args.cfg[0]

    metadatas = load_dpg_prompts(args.prompt_dir)
    image_root = osp.join(args.outdir, args.image_dir_name)
    sample_root = osp.join(args.outdir, 'samples')
    os.makedirs(image_root, exist_ok=True)
    if args.save_individual:
        os.makedirs(sample_root, exist_ok=True)

    if args.model_type == 'sdxl':
        from diffusers import DiffusionPipeline
        base = DiffusionPipeline.from_pretrained(
            'stabilityai/stable-diffusion-xl-base-1.0',
            torch_dtype=torch.float16,
            variant='fp16',
            use_safetensors=True,
        ).to('cuda')
        refiner = DiffusionPipeline.from_pretrained(
            'stabilityai/stable-diffusion-xl-refiner-1.0',
            text_encoder_2=base.text_encoder_2,
            vae=base.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant='fp16',
        ).to('cuda')
    elif args.model_type == 'sd3':
        from diffusers import StableDiffusion3Pipeline
        pipe = StableDiffusion3Pipeline.from_pretrained(
            'stabilityai/stable-diffusion-3-medium-diffusers',
            torch_dtype=torch.float16,
        )
        pipe = pipe.to('cuda')
    elif args.model_type == 'pixart_sigma':
        from diffusers import PixArtSigmaPipeline
        pipe = PixArtSigmaPipeline.from_pretrained(
            'PixArt-alpha/PixArt-Sigma-XL-2-1024-MS',
            torch_dtype=torch.float16,
        ).to('cuda')
    elif args.model_type == 'flux_1_dev':
        from diffusers import FluxPipeline
        pipe = FluxPipeline.from_pretrained(
            'black-forest-labs/FLUX.1-dev',
            torch_dtype=torch.bfloat16,
        ).to('cuda')
    elif args.model_type == 'flux_1_dev_schnell':
        from diffusers import FluxPipeline
        pipe = FluxPipeline.from_pretrained(
            'black-forest-labs/FLUX.1-schnell',
            torch_dtype=torch.bfloat16,
        ).to('cuda')
    elif 'infinity' in args.model_type:
        text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
        vae = load_visual_tokenizer(args)
        infinity = load_transformer(vae, args)
        
        should_apply_quant = bool(getattr(args, 'quantize_weight', 0)) or bool(getattr(args, 'quantize_act', 0))
        if should_apply_quant:
            replace_linear_modules(infinity, args)

        if args.rewrite_prompt:
            from tools.prompt_rewriter import PromptRewriter
            prompt_rewriter = PromptRewriter(system='', few_shot_history=[])

    save_metadata_file_path = osp.join(args.outdir, 'metadata.jsonl')
    existing_metadatas = {}
    if osp.isfile(save_metadata_file_path):
        try:
            with open(save_metadata_file_path, 'r', encoding='utf-8') as fp:
                existing_items = json.load(fp)
            existing_metadatas = {
                item.get('id'): item
                for item in existing_items
                if item.get('id') is not None and complete_grid_path(item)
            }
            print(f'Resume: found {len(existing_metadatas)} complete DPG prompts in {save_metadata_file_path}')
        except Exception as exc:
            print(f'Resume: ignoring unreadable metadata file {save_metadata_file_path}: {exc}')

    save_metadatas = []
    for index, metadata in enumerate(metadatas):
        seed_everything(args.seed)
        prompt_id = metadata['id']
        original_prompt = metadata['prompt']
        grid_image_path = osp.join(image_root, f'{prompt_id}.{args.image_ext}')
        print(f"Prompt ({index: >4}/{len(metadatas)}): id={prompt_id}, prompt='{original_prompt}'")

        if prompt_id in existing_metadatas:
            print(f'Skip prompt {prompt_id}, complete DPG grid already exists.')
            save_metadatas.append(existing_metadatas[prompt_id])
            continue
        if osp.isfile(grid_image_path) and osp.getsize(grid_image_path) > 0:
            print(f'Skip prompt {prompt_id}, found existing grid image: {grid_image_path}')
            save_metadatas.append(
                grid_metadata(
                    metadata,
                    grid_image_path=grid_image_path,
                    sample_image_paths=[],
                    tile_resolution=args.grid_resolution,
                    prompt_for_generation=original_prompt,
                )
            )
            save_metadata_atomic(save_metadata_file_path, save_metadatas)
            continue

        tau = args.tau
        cfg = args.cfg
        prompt = original_prompt
        if args.rewrite_prompt:
            refined_prompt = prompt_rewriter.rewrite(prompt)
            input_key_val = extract_key_val(refined_prompt)
            prompt = input_key_val['prompt']
            print(f'prompt: {prompt}, refined_prompt: {refined_prompt}')

        sample_images = []
        sample_image_paths = []
        for sample_i in range(args.n_samples):
            if args.model_type == 'sdxl':
                image = base(
                    prompt=prompt,
                    num_inference_steps=40,
                    denoising_end=0.8,
                    output_type='latent',
                ).images
                image = refiner(
                    prompt=prompt,
                    num_inference_steps=40,
                    denoising_start=0.8,
                    image=image,
                ).images[0]
            elif args.model_type == 'sd3':
                image = pipe(
                    prompt,
                    negative_prompt='',
                    num_inference_steps=28,
                    guidance_scale=7.0,
                    num_images_per_prompt=1,
                ).images[0]
            elif args.model_type == 'flux_1_dev':
                image = pipe(
                    prompt,
                    height=1024,
                    width=1024,
                    guidance_scale=3.5,
                    num_inference_steps=50,
                    max_sequence_length=512,
                    num_images_per_prompt=1,
                ).images[0]
            elif args.model_type == 'flux_1_dev_schnell':
                image = pipe(
                    prompt,
                    height=1024,
                    width=1024,
                    guidance_scale=0.0,
                    num_inference_steps=4,
                    max_sequence_length=256,
                    generator=torch.Generator('cpu').manual_seed(0),
                ).images[0]
            elif args.model_type == 'pixart_sigma':
                image = pipe(prompt).images[0]
            elif 'infinity' in args.model_type:
                h_div_w_template = 1.000
                scale_schedule = dynamic_resolution_h_w[h_div_w_template][args.pn]['scales']
                scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
                image = gen_one_img(
                    infinity,
                    vae,
                    text_tokenizer,
                    text_encoder,
                    prompt,
                    tau_list=tau,
                    cfg_sc=3,
                    cfg_list=cfg,
                    scale_schedule=scale_schedule,
                    cfg_insertion_layer=[args.cfg_insertion_layer],
                    vae_type=args.vae_type,
                    sampling_per_bits=args.sampling_per_bits,
                    enable_positive_prompt=args.enable_positive_prompt,
                )
            else:
                raise ValueError(f'Unsupported model_type={args.model_type}')

            image_arr = image_to_bgr_array(image)

            sample_image_path = ''
            if args.save_individual:
                prompt_sample_path = osp.join(sample_root, prompt_id)
                os.makedirs(prompt_sample_path, exist_ok=True)
                sample_image_path = osp.join(prompt_sample_path, f'{prompt_id}_{sample_i}.{args.image_ext}')
                if not cv2.imwrite(sample_image_path, image_arr):
                    raise RuntimeError(f'failed to write image: {sample_image_path}')
                sample_image_paths.append(sample_image_path)

            sample_images.append(image_arr)

        tile_resolution = args.grid_resolution
        if tile_resolution <= 0:
            tile_resolution = int(sample_images[0].shape[0])
        grid = make_dpg_grid(sample_images, tile_resolution)
        if not cv2.imwrite(grid_image_path, grid):
            raise RuntimeError(f'failed to write DPG grid image: {grid_image_path}')
        print(grid_image_path)
        save_metadatas.append(
            grid_metadata(
                metadata,
                grid_image_path=grid_image_path,
                sample_image_paths=sample_image_paths,
                tile_resolution=tile_resolution,
                prompt_for_generation=prompt,
            )
        )
        save_metadata_atomic(save_metadata_file_path, save_metadatas)
