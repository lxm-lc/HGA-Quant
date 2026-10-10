import argparse
import copy
import datetime
import os
import random
import time

import numpy as np
import torch
import torchvision
from PIL import Image
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)

from hart.modules.models.transformer import HARTForT2I
from hart.utils import default_prompts, encode_prompts, llm_system_prompt, safety_check
from quant import apply_quantization


def save_images(sample_imgs, sample_folder_dir, store_separately, prompts):
    if not store_separately and len(sample_imgs) > 1:
        grid = torchvision.utils.make_grid(sample_imgs, nrow=12)
        grid_np = grid.to(torch.float16).permute(1, 2, 0).mul_(255).cpu().numpy()

        os.makedirs(sample_folder_dir, exist_ok=True)
        grid_np = Image.fromarray(grid_np.astype(np.uint8))
        grid_np.save(os.path.join(sample_folder_dir, f"sample_images.png"))
        print(f"Example images are saved to {sample_folder_dir}")
    else:
        # bs, 3, r, r
        sample_imgs_np = sample_imgs.mul_(255).cpu().numpy()
        num_imgs = sample_imgs_np.shape[0]
        os.makedirs(sample_folder_dir, exist_ok=True)
        for img_idx in range(num_imgs):
            cur_img = sample_imgs_np[img_idx]
            cur_img = cur_img.transpose(1, 2, 0).astype(np.uint8)
            cur_img_store = Image.fromarray(cur_img)
            cur_img_store.save(os.path.join(sample_folder_dir, f"{img_idx:06d}.png"))
            print(f"Image {img_idx} saved.")

    with open(os.path.join(sample_folder_dir, "prompt.txt"), "w") as f:
        f.write("\n".join(prompts))


class HARTSampler:
    """Load HART and its current quantization path once, then reuse it."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda")

        model = AutoModel.from_pretrained(args.model_path)
        model = model.to(self.device)
        model.eval()

        if args.use_ema:
            ema_model = copy.deepcopy(model)
            ema_model.load_state_dict(
                torch.load(os.path.join(args.model_path, "ema_model.bin"))
            )
            self.infer_model = ema_model
        else:
            self.infer_model = model

        self.quant_act_scheduler = None
        self.text_tokenizer = None
        self.text_model = None
        self.use_safety_checker = bool(getattr(args, "use_safety_checker", 1))
        self.safety_checker_tokenizer = None
        self.safety_checker_model = None
        self._prepared = False

    def _prepare_once(self):
        if self._prepared:
            return

        apply_quantization(self.infer_model, self.args)
        self.quant_act_scheduler = getattr(
            self.infer_model, "coarse_to_fine_act_scheduler", None
        )

        self.text_tokenizer = AutoTokenizer.from_pretrained(self.args.text_model_path)
        self.text_model = AutoModel.from_pretrained(self.args.text_model_path).to(
            self.device
        )
        self.text_model.eval()

        self.safety_checker_tokenizer = None
        self.safety_checker_model = None
        if self.use_safety_checker:
            self.safety_checker_tokenizer = AutoTokenizer.from_pretrained(
                self.args.shield_model_path
            )
            self.safety_checker_model = AutoModelForCausalLM.from_pretrained(
                self.args.shield_model_path,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            ).to(self.device)
        self._prepared = True

    def generate(self, prompts, generation_seed=None):
        prompts = list(prompts)
        if generation_seed is None:
            generation_seed = self.args.seed

        self._prepare_once()

        if self.use_safety_checker:
            for idx, prompt in enumerate(prompts):
                if safety_check.is_dangerous(
                    self.safety_checker_tokenizer,
                    self.safety_checker_model,
                    prompt,
                ):
                    prompts[idx] = random.sample(default_prompts, 1)[0]
                    print(
                        f"Detected Unsafe prompt with index {idx}, "
                        "will replace by one of default prompts."
                    )

        start_time = time.time()
        with torch.inference_mode():
            with torch.autocast(
                "cuda", enabled=True, dtype=torch.float16, cache_enabled=True
            ):
                (
                    context_tokens,
                    context_mask,
                    context_position_ids,
                    context_tensor,
                ) = encode_prompts(
                    prompts,
                    self.text_model,
                    self.text_tokenizer,
                    self.args.max_token_length,
                    llm_system_prompt,
                    self.args.use_llm_system_prompt,
                )

                output_imgs = self.infer_model.autoregressive_infer_cfg(
                    B=context_tensor.size(0),
                    label_B=context_tensor,
                    cfg=self.args.cfg,
                    g_seed=generation_seed,
                    more_smooth=self.args.more_smooth,
                    context_position_ids=context_position_ids,
                    context_mask=context_mask,
                    quant_act_scheduler=self.quant_act_scheduler,
                )

        total_time = time.time() - start_time
        print(f"Generate {len(prompts)} images take {total_time:2f}s.")
        return output_imgs, prompts


def main(args):
    sampler = HARTSampler(args)
    if args.prompt:
        prompts = [args.prompt]
    elif args.prompt_list:
        prompts = args.prompt_list
    else:
        print(
            "No prompt is provided. Will randomly sample 4 prompts from default prompts."
        )
        prompts = random.sample(default_prompts, 4)

    output_imgs, prompts = sampler.generate(prompts)

    save_images(
        output_imgs.clone(), args.sample_folder_dir, args.store_seperately, prompts
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        help="The path to HART model.",
        default="pretrained_models/HART-1024",
    )
    parser.add_argument(
        "--text_model_path",
        type=str,
        help="The path to text model, we employ Qwen2-VL-1.5B-Instruct by default.",
        default="Qwen2-VL-1.5B-Instruct",
    )
    parser.add_argument(
        "--shield_model_path",
        type=str,
        help="The path to shield model, we employ ShieldGemma-2B by default.",
        default="pretrained_models/shieldgemma-2b",
    )
    parser.add_argument("--prompt", type=str, help="A single prompt.", default="")
    parser.add_argument("--prompt_list", type=list[str], default=[])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--use_ema", type=bool, default=True)
    parser.add_argument("--max_token_length", type=int, default=300)
    parser.add_argument("--use_llm_system_prompt", type=bool, default=True)
    parser.add_argument(
        "--cfg", type=float, help="Classifier-free guidance scale.", default=4.5
    )
    parser.add_argument(
        "--more_smooth",
        type=bool,
        help="Turn on for more visually smooth samples.",
        default=True,
    )
    parser.add_argument(
        "--sample_folder_dir",
        type=str,
        help="The folder where the image samples are stored",
        default="samples/",
    )
    parser.add_argument(
        "--store_seperately",
        help="Store image samples in a grid or separately, set to False by default.",
        action="store_true",
    )
    parser.add_argument("--quantize_weight", type=int, default=0, choices=[0, 1])
    parser.add_argument("--quantize_act", type=int, default=0, choices=[0, 1])
    parser.add_argument("--weight_bits", type=int, default=4, choices=[2, 4, 6, 8])
    parser.add_argument("--act_bits", type=int, default=6, choices=[2, 4, 6, 8])
    parser.add_argument("--use_hadamard", type=int, default=0, choices=[0, 1])
    parser.add_argument("--use_tbgr", type=int, default=0, choices=[0, 1])
    parser.add_argument("--tbgr_outlier_factor", type=float, default=1.0)
    parser.add_argument("--tbgr_outlier_sigma", type=float, default=3.0)
    parser.add_argument(
        "--mixed_act_mode",
        type=str,
        default="default",
        choices=["default", "coarse_to_fine"],
    )
    parser.add_argument(
        "--ctf_act_policy",
        type=str,
        default="early_high",
        choices=["early_high", "late_high", "u_shape"],
    )
    parser.add_argument("--ctf_low_bits", type=int, default=4, choices=[2, 4, 6, 8])
    parser.add_argument("--ctf_high_bits", type=int, default=8, choices=[2, 4, 6, 8])
    parser.add_argument("--ctf_target_avg_bits", type=float, default=6.0)
    parser.add_argument("--ctf_u_late_high_count", type=int, default=1)
    args = parser.parse_args()

    main(args)
