import argparse
import contextlib
import json
from pathlib import Path
from collections import Counter

import torch
from PIL import Image

from models import build_vae_var
from quant import apply_quantization


DEFAULT_PATCH_NUMS = '1_2_3_4_5_6_8_10_13_16'


def parse_patch_nums(text: str):
    return tuple(int(item) for item in str(text).replace('-', '_').split('_') if item)


def default_var_ckpt(depth: int) -> str:
    return f'pretrained_models/var_d{int(depth)}.pth'


def build_label_schedule(args):
    label_mode = str(args.label_mode).lower().strip()
    if label_mode == 'random':
        total = int(args.num_images)
        if total <= 0:
            raise ValueError('--num_images must be positive when --label_mode random')
        return total, None, {}

    if label_mode == 'fixed':
        if args.label is None:
            raise ValueError('--label must be specified when --label_mode fixed')
        total = int(args.num_images)
        if total <= 0:
            raise ValueError('--num_images must be positive when --label_mode fixed')
        labels = [int(args.label)] * total
        return total, labels, {int(args.label): total}

    if label_mode == 'balanced':
        if args.class_start < 0 or args.class_end >= args.num_classes or args.class_end < args.class_start:
            raise ValueError(
                f'invalid class range [{args.class_start}, {args.class_end}] for num_classes={args.num_classes}'
            )
        if args.samples_per_class <= 0:
            raise ValueError('--samples_per_class must be positive when --label_mode balanced')
        labels = []
        for class_id in range(int(args.class_start), int(args.class_end) + 1):
            labels.extend([class_id] * int(args.samples_per_class))
        total = len(labels)
        if int(args.num_images) not in (0, total):
            raise ValueError(
                f'balanced ImageNet schedule expects {total} images '
                f'({args.class_end - args.class_start + 1} classes x {args.samples_per_class}), '
                f'got --num_images {args.num_images}. Use --num_images 0 to derive it automatically.'
            )
        return total, labels, dict(Counter(labels))

    raise ValueError(f'unsupported label_mode={args.label_mode!r}')


def build_argparser():
    parser = argparse.ArgumentParser(description='Sample ImageNet images with VAR and optional fake quantization.')

    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--num_images', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--label', type=int, default=None)
    parser.add_argument('--label_mode', type=str, default='random', choices=['random', 'fixed', 'balanced'])
    parser.add_argument('--samples_per_class', type=int, default=50)
    parser.add_argument('--class_start', type=int, default=0)
    parser.add_argument('--class_end', type=int, default=999)
    parser.add_argument('--cfg', type=float, default=1.5)
    parser.add_argument('--top_k', type=int, default=900)
    parser.add_argument('--top_p', type=float, default=0.96)
    parser.add_argument('--more_smooth', type=int, default=0, choices=[0, 1])

    parser.add_argument('--depth', type=int, default=16)
    parser.add_argument('--pn', type=str, default=DEFAULT_PATCH_NUMS)
    parser.add_argument('--vae_ckpt', type=str, default='pretrained_models/vae_ch160v4096z32.pth')
    parser.add_argument('--var_ckpt', type=str, default='')
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--fp16', type=int, default=1, choices=[0, 1])
    parser.add_argument('--fuse', type=int, default=1, choices=[0, 1])

    parser.add_argument('--quantize_weight', type=int, default=0, choices=[0, 1])
    parser.add_argument('--quantize_act', type=int, default=0, choices=[0, 1])
    parser.add_argument('--quant_seed', type=int, default=0)
    parser.add_argument('--weight_bits', type=int, default=4, choices=[2, 4, 6, 8])
    parser.add_argument('--act_bits', type=int, default=6, choices=[2, 4, 6, 8])
    parser.add_argument('--use_hadamard', type=int, default=1, choices=[0, 1])
    parser.add_argument('--use_tbgr', type=int, default=1, choices=[0, 1])
    parser.add_argument('--tbgr_outlier_factor', type=float, default=1.0)
    parser.add_argument('--tbgr_outlier_sigma', type=float, default=3.0)
    parser.add_argument(
        '--mixed_act_mode',
        type=str,
        default='default',
        choices=['default', 'coarse_to_fine'],
    )

    parser.add_argument('--ctf_act_policy', type=str, default='early_high', choices=['early_high', 'late_high', 'u_shape'])
    parser.add_argument('--ctf_low_bits', type=int, default=4, choices=[2, 4, 6, 8])
    parser.add_argument('--ctf_high_bits', type=int, default=8, choices=[2, 4, 6, 8])
    parser.add_argument('--ctf_target_avg_bits', type=float, default=6.0)
    parser.add_argument('--ctf_u_late_high_count', type=int, default=1)

    return parser


def save_images(imgs: torch.Tensor, out_dir: Path, start_index: int, labels=None):
    imgs = (
        imgs.detach()
        .float()
        .clamp(0, 1)
        .mul(255)
        .round()
        .to(torch.uint8)
        .permute(0, 2, 3, 1)
        .cpu()
        .numpy()
    )
    for local_idx, arr in enumerate(imgs):
        image_index = start_index + local_idx
        Image.fromarray(arr).save(out_dir / f'{image_index:05d}.png')
    return len(imgs)


def main():
    parser = build_argparser()
    args = parser.parse_args()
    args.mixed_act_mode = args.mixed_act_mode
    total, label_schedule, label_histogram = build_label_schedule(args)
    args.num_images = total

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high')

    patch_nums = parse_patch_nums(args.pn)
    var_ckpt = args.var_ckpt or default_var_ckpt(args.depth)
    print(f'[Sample] out_dir={out_dir}')
    print(f'[Sample] depth={args.depth}, patch_nums={patch_nums}, var_ckpt={var_ckpt}')
    print(f'[Sample] label_mode={args.label_mode}, num_images={args.num_images}, batch_size={args.batch_size}')

    vae, var = build_vae_var(
        device=args.device,
        patch_nums=patch_nums,
        num_classes=args.num_classes,
        depth=args.depth,
        flash_if_available=bool(args.fuse),
        fused_if_available=bool(args.fuse),
    )
    vae.load_state_dict(torch.load(args.vae_ckpt, map_location='cpu'), strict=True)
    var.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=True)
    vae.eval()
    var.eval()
    for module in (vae, var):
        for param in module.parameters():
            param.requires_grad_(False)

    quant_summary = None
    cuda_devices = []
    quant_device = torch.device(args.device)
    if quant_device.type == 'cuda' and torch.cuda.is_available():
        cuda_devices = [torch.cuda.current_device() if quant_device.index is None else quant_device.index]
    with torch.random.fork_rng(devices=cuda_devices, enabled=True):
        torch.manual_seed(args.quant_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.quant_seed)
        quant_summary = apply_quantization(var, args=args)
    if quant_summary is not None:
        quant_summary['quant_seed'] = args.quant_seed
        print('[Quant] ' + json.dumps({k: v for k, v in quant_summary.items() if k != 'init_act_bits_by_module'}, indent=2, sort_keys=True))

    official_imagenet256 = (
        args.label_mode == 'balanced'
        and args.num_classes == 1000
        and args.class_start == 0
        and args.class_end == 999
        and args.samples_per_class == 50
        and args.num_images == 50000
    )
    labels_file = None
    if label_schedule is not None:
        labels_file = out_dir / 'labels.json'
        with open(labels_file, 'w', encoding='utf-8') as fp:
            json.dump(label_schedule, fp)

    metadata = {
        'args': vars(args),
        'patch_nums': patch_nums,
        'var_ckpt': var_ckpt,
        'quant_summary': quant_summary,
        'label_mode': args.label_mode,
        'label_histogram': {str(k): int(v) for k, v in sorted(label_histogram.items())},
        'labels_file': None if labels_file is None else labels_file.name,
        'official_imagenet256': official_imagenet256,
        'sampling_protocol': (
            'imagenet256_1000_classes_50_per_class'
            if official_imagenet256
            else f'{args.label_mode}_sample'
        ),
    }
    with open(out_dir / 'metadata.json', 'w', encoding='utf-8') as fp:
        json.dump(metadata, fp, indent=2, sort_keys=True)

    def amp_context():
        if args.fp16 and str(args.device).startswith('cuda'):
            return torch.autocast('cuda', dtype=torch.float16)
        return contextlib.nullcontext()

    batch_size = max(1, int(args.batch_size))
    saved = 0
    while saved < total:
        cur_bs = min(batch_size, total - saved)
        seed = args.seed + saved
        label_B = None
        if label_schedule is not None:
            label_B = torch.tensor(label_schedule[saved:saved + cur_bs], dtype=torch.long, device=args.device)
        with torch.inference_mode(), amp_context():
            imgs = var.autoregressive_infer_cfg(
                B=cur_bs,
                label_B=label_B if label_B is not None else args.label,
                g_seed=seed,
                cfg=args.cfg,
                top_k=args.top_k,
                top_p=args.top_p,
                more_smooth=bool(args.more_smooth),
            )
        saved += save_images(imgs, out_dir, saved, labels=None if label_schedule is None else label_schedule[saved:saved + cur_bs])
        print(f'[Sample] saved {saved}/{total}', flush=True)

    print(f'[Sample] done: {saved} images in {out_dir}')


if __name__ == '__main__':
    main()
