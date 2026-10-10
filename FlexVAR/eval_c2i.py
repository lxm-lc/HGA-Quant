import os
import random
import argparse
import math
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
from models import build_vae_var
import time
from utils.evaluations.c2i.img2npy import create_npz_from_sample_folder
from quant import apply_linear_quantization, build_coarse_to_fine_act_scheduler
from utils.image_utils import save_tensor_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--var_ckpt", type=str, default="out/VARd20/40epo/ar-ckpt-epo35.pth")
    parser.add_argument("--vae_ckpt", type=str, default="pretrained/FlexVAE.pth")
    parser.add_argument("--cfg", type=float, default=4, )
    parser.add_argument("--top_k", type=int, default=900, )
    parser.add_argument("--maxpn", type=int, default=16, )
    parser.add_argument("--depth", type=int, default=24, )
    parser.add_argument("--infer_patch_nums", type=str, default="1_2_3_4_5_6_8_10_13_16", )
    parser.add_argument("--output_root", type=str, default="output")
    parser.add_argument("--exp_name", type=str, default="auto")
    parser.add_argument("--max_images", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quantize_weight", type=int, default=0, choices=[0, 1])
    parser.add_argument("--quantize_act", type=int, default=0, choices=[0, 1])
    parser.add_argument("--weight_bits", type=int, default=4, choices=[2, 4, 6, 8])
    parser.add_argument("--act_bits", type=int, default=6, choices=[2, 4, 6, 8])
    parser.add_argument("--use_hadamard", type=int, default=1, choices=[0, 1])
    parser.add_argument("--use_tbgr", type=int, default=1, choices=[0, 1])
    parser.add_argument("--tbgr_outlier_factor", type=float, default=8.0)
    parser.add_argument("--tbgr_outlier_sigma", type=float, default=3.0)
    parser.add_argument("--mixed_act_mode", type=str, default="default", choices=["default", "coarse_to_fine"])
    parser.add_argument("--ctf_act_policy", type=str, default="early_high", choices=["early_high", "late_high", "u_shape"])
    parser.add_argument("--ctf_low_bits", type=int, default=4, choices=[2, 4, 6, 8])
    parser.add_argument("--ctf_high_bits", type=int, default=8, choices=[2, 4, 6, 8])
    parser.add_argument("--ctf_target_avg_bits", type=float, default=6.0)
    parser.add_argument("--ctf_u_late_high_count", type=int, default=1)

    args = parser.parse_args()

    if args.mixed_act_mode == "coarse_to_fine" and not args.quantize_act:
        raise ValueError(
            "coarse_to_fine quantization requires --quantize_act 1."
        )

    MODEL_DEPTH = args.depth    # TODO: =====> please specify MODEL_DEPTH <=====
    assert MODEL_DEPTH in {16, 20, 24, 30}
    infer_patch_nums = tuple(map(int, args.infer_patch_nums.replace('-', '_').split('_')))
    if args.exp_name == 'auto':
        args.exp_name = f'd{MODEL_DEPTH}_cfg{args.cfg}_shape{infer_patch_nums[-1]*16}_{len(infer_patch_nums)}step_maxpn{args.maxpn}'
    save_root = os.path.join(args.output_root, args.exp_name)
    os.makedirs(save_root, exist_ok=True)

    dist.init_process_group(backend='nccl')
    global_rank = dist.get_rank()
    device = global_rank % torch.cuda.device_count()
    if global_rank == 0:
        print(args)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.set_device(device)


    # download checkpoint
    vae_ckpt = args.vae_ckpt
    var_ckpt = args.var_ckpt

    # build vae, var
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vae, var = build_vae_var(
        V=8912,  # 8912 8192
        Cvae=32, device=device, num_classes=1000, depth=MODEL_DEPTH, shared_aln=False,
        vae_ckpt = vae_ckpt,
        flash_if_available=True,
        fused_if_available=True,
    )
    # load checkpoints
    ckpt = torch.load(var_ckpt, map_location='cpu')
    if 'trainer' in ckpt.keys():
        ckpt = ckpt['trainer']['var_wo_ddp']
    old_params = var.state_dict()
    ckpt["attn_bias_for_masking"] = old_params["attn_bias_for_masking"]
    var.load_state_dict(ckpt, strict=True)
    vae.eval(), var.eval()
    for p in vae.parameters(): p.requires_grad_(False)
    for p in var.parameters(): p.requires_grad_(False)
    var.update_patch_related(infer_patch_nums)

    num_total_images = max(1, int(args.max_images))
    repeat = math.ceil(num_total_images / 1000)
    data_list = ([i for i in range(1000)] * repeat)[:num_total_images]
    random.shuffle(data_list)

    if args.quantize_weight or args.quantize_act:
        apply_linear_quantization(
            var,
            weight_quant=bool(args.quantize_weight),
            act_quant=bool(args.quantize_act),
            weight_bits=args.weight_bits,
            act_bits=args.act_bits,
            ctf_low_bits=args.ctf_low_bits,
            use_hadamard=bool(args.use_hadamard),
            use_tbgr=bool(args.use_tbgr),
            mixed_act_mode=args.mixed_act_mode,
            tbgr_outlier_factor=args.tbgr_outlier_factor,
            tbgr_outlier_sigma=args.tbgr_outlier_sigma,
            hadamard_seed=args.seed,
        )
        if args.mixed_act_mode == "coarse_to_fine":
            var.set_quant_act_scheduler(build_coarse_to_fine_act_scheduler(var, args))
        else:
            var.set_quant_act_scheduler(None)
        var.eval()
    print('prepare finished.')


    ############################# 2. Sample with classifier-free guidance

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # run faster
    tf32 = True
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.set_float32_matmul_precision('high' if tf32 else 'highest')

    data_sublist = data_list[global_rank::dist.get_world_size()]
    dataset = [data_sublist[i:i + args.batch_size] for i in range(0, len(data_sublist), args.batch_size)]
    with torch.inference_mode():
        with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):    # using bfloat16 can be faster

            for num, c in enumerate(dataset):
                label_B: torch.LongTensor = torch.tensor(c, device=device)

                t1 = time.time()
                # sample
                B = len(c)
                recon_B3HW = var.autoregressive_infer_cfg(
                    vqvae=vae,
                    B=B,
                    label_B=label_B,
                    infer_patch_nums=infer_patch_nums,
                    cfg=args.cfg,
                    top_k=args.top_k,
                    top_p=0.95,
                    g_seed=None,
                    more_smooth=False,
                    max_pn=args.maxpn,
                )

                for i in range(len(c)):
                    save_tensor_image(recon_B3HW[i], f"{save_root}/rank{global_rank}-{num * args.batch_size + i}.png", value_range=(-1, 1))
                t2 = time.time()

                if num % 5 == 0:
                    print(f"curr step {num}, total-step{len(dataset)}, processing time: {t2-t1}s, etc. {(t2-t1)*(len(dataset)-num)}")



    if global_rank == 0 and num_total_images == 50000:
        time.sleep(1*60)
        create_npz_from_sample_folder(save_root)

        print('done !')
        print("save_root:", save_root)
    elif global_rank == 0:
        print('done !')
        print("save_root:", save_root)
        
