# Author: Qian Liu
# Email: liu.qian.pro@gmail.com

import argparse
import torch
import open_clip
from PIL import Image
from pathlib import Path
import random, glob, os
from tqdm import tqdm
import torch.nn.functional as F


def extract_clip_text_features(args, scene_classes):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    model, _, _ = open_clip.create_model_and_transforms(
        model_name=args.clip_model,
        pretrained=args.pretrained,
        device=device
    )
    model.eval()

    tokenizer = open_clip.get_tokenizer(args.clip_model)
    text = tokenizer(scene_classes).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features = F.normalize(text_features, dim=1)

    # save the extracted feature
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)

    text_features = text_features.detach().cpu()  # Move to CPU for saving
    print(f"Extracted text features with shape {text_features.shape} for classes: {scene_classes}")
    print(f"Saving text features to {save_path}/scene_classes.pt")
    # Save the text features
    torch.save(text_features, f'{save_path}/scene_classes.pt')


def extract_clip_features(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=args.clip_model,
        pretrained=args.pretrained,
        force_quick_gelu=True,
        device=device
    )
    model.eval()

    image_paths = sorted(glob.glob(f'{args.images_path}/*'))

    for image_path in tqdm(image_paths):

        # down_sample = 8
        image_path = Path(image_path)
        image = Image.open(image_path)
        # image = image.resize((image.width//down_sample, image.height//down_sample))

        patch_sizes = [min(image.size)//5, min(image.size)//7, min(image.size)//10]

        image_feature = []

        # loop over all the scale
        for patch_size in patch_sizes:
            stride = patch_size // 4
            patches = []
            idxes = []
            # loop to get all the patches
            for x_idx in range((image.height-patch_size)//stride + 1 + int((image.height-patch_size)%stride>0)):
                start_x = x_idx * stride
                for y_idx in range((image.width-patch_size)//stride + 1 + int((image.width-patch_size)%stride>0)):
                    start_y = y_idx * stride
                    # add randomness
                    (left, upper, right, lower) = (
                        max(start_y-random.randint(0, patch_size // 4), 0),
                        max(start_x-random.randint(0, patch_size // 4), 0),
                        min(start_y+patch_size+random.randint(0, patch_size // 4), image.width),
                        min(start_x+patch_size+random.randint(0, patch_size // 4), image.height)
                    )
                    patches.append(preprocess(image.crop((left, upper, right, lower))))
                    idxes.append((left, upper, right, lower))

            # get clip embedding
            count = torch.zeros((1, 1, image.height, image.width)).to(device)
            sum_feature = torch.zeros((1, 512, image.height, image.width)).to(device)

            with torch.no_grad():
                chunk_size = 8
                for chunk_idx in range(len(patches)//chunk_size + int(len(patches)%chunk_size>0)):
                    patch_chunk = torch.stack(patches[chunk_idx*chunk_size : (chunk_idx+1)*chunk_size]).to(device)
                    patch_chunk_feature = model.encode_image(patch_chunk)
                    for i in range(chunk_size):
                        patch_idx = chunk_idx*chunk_size + i
                        if patch_idx >= len(idxes): break

                        sum_feature[:, :, idxes[patch_idx][1]:idxes[patch_idx][3], idxes[patch_idx][0]:idxes[patch_idx][2]] += \
                            patch_chunk_feature[i:i+1, :, None, None]
                        count[:, :, idxes[patch_idx][1]:idxes[patch_idx][3], idxes[patch_idx][0]:idxes[patch_idx][2]] += 1

                image_feature.append(sum_feature / count)

        image_feature = torch.cat(image_feature).detach().cpu() # [scale, D, height, width]
        print(f"Extracted features for {image_path.name} with shape {image_feature.shape}")

        # save the extracted feature
        save_path = args.save_path
        os.makedirs(save_path, exist_ok=True)
        print(f"Saving image features to {save_path}/{image_path.stem}.pt")
        torch.save(image_feature, f'{save_path}/{image_path.stem}.pt')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract CLIP pixel features from images")
    parser.add_argument("--images_path", type=str)
    parser.add_argument("--clip_model", type=str, default="ViT-B-16")
    parser.add_argument("--pretrained", type=str, default="dfn2b")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--save_path", type=str)
    args = parser.parse_args()

    extract_clip_features(args)