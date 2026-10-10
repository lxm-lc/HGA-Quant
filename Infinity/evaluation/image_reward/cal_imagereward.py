import os
os.environ['XFORMERS_FORCE_DISABLE_TRITON'] = '1'
import os.path as osp
import json
import argparse

import numpy as np
import ImageReward as RM


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_file", type=str, default="")
    default_download_root = osp.expanduser("~/.cache/ImageReward")
    parser.add_argument(
        "--model_path",
        type=str,
        default=osp.join(default_download_root, "ImageReward.pt"),
        help="Local ImageReward checkpoint. Falls back to the package download path if missing.",
    )
    parser.add_argument(
        "--med_config",
        type=str,
        default=osp.join(default_download_root, "med_config.json"),
        help="Local ImageReward med_config.json. Falls back to the package download path if missing.",
    )
    parser.add_argument(
        "--download_root",
        type=str,
        default=default_download_root,
        help="Cache directory used for the fallback ImageReward/CLIP downloads.",
    )
    args = parser.parse_args()

    model_path = osp.abspath(osp.expanduser(args.model_path))
    med_config = osp.abspath(osp.expanduser(args.med_config))
    download_root = osp.abspath(osp.expanduser(args.download_root))

    if osp.isfile(model_path) and osp.isfile(med_config):
        print(f"Loading ImageReward from local files: {model_path}, {med_config}")
        image_reward_model = RM.load(
            model_path,
            med_config=med_config,
        )
    else:
        missing = []
        if not osp.isfile(model_path):
            missing.append(model_path)
        if not osp.isfile(med_config):
            missing.append(med_config)
        print(
            "Local ImageReward files are incomplete; falling back to the package "
            f"download path. Missing: {missing}"
        )
        image_reward_model = RM.load(
            "ImageReward-v1.0",
            download_root=download_root,
        )
    clip_model = RM.load_score("CLIP", download_root=download_root)

    with open(args.meta_file, 'r') as f:
        meta_infos = json.load(f)
    
    average_image_reward = []
    average_clip_scores = []
    for meta in meta_infos:
        image_paths = meta['gen_image_paths']
        prompt = meta['prompt']
        image_rewards = image_reward_model.score(prompt, image_paths)
        _, clip_scores = clip_model.inference_rank(prompt, image_paths)
        # average_image_reward.extend(image_rewards)
        # average_clip_scores.extend(clip_scores)
        if isinstance(image_rewards, float):
            average_image_reward.append(image_rewards)
            average_clip_scores.append(clip_scores)
        else:
            average_image_reward.extend(image_rewards)
            average_clip_scores.extend(clip_scores)
        print(f'Average Image Reward of {len(meta_infos)} prompt and {len(average_image_reward)} images is {np.mean(average_image_reward):.4f}, Average CLIP Score is {np.mean(average_clip_scores):.4f}')
    print(f'Average Image Reward of {len(meta_infos)} prompt and {len(average_image_reward)} images is {np.mean(average_image_reward):.4f}, Average CLIP Score is {np.mean(average_clip_scores):.4f}')
    save_file = osp.join(osp.dirname(args.meta_file), 'image_reward_res.json')
    with open(save_file, 'w') as f:
        json.dump({
            'prompts': len(meta_infos),
            'images': len(average_image_reward),
            'average_image_reward': np.mean(average_image_reward),
            'average_clip_scores': np.mean(average_clip_scores)
        }, f)
    print(f'Save to {save_file}')
