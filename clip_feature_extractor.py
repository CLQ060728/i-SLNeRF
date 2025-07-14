# Author: Qian Liu
# Email: liu.qian.pro@gmail.com

import argparse
import torch
import open_clip
from PIL import Image
from PIL.Image import Resampling
from pathlib import Path
import random, glob, os
from tqdm import tqdm
import torch.nn.functional as F
import json


prompt_templates = ["a photo of {}", "a picture of {}", "a rendering of {}", "an image of {}",
                    "a scene of {}", "an outdoor scene of {}", "a photo of {} scene",
                    "an image of {} scene", "a picture of {} scene", "a rendering of {} scene"]

# adopted in our i-SLNeRF paper
def get_clip_text_features(args):
    """
    Get CLIP text features for a list of scene classes.
    :param args: Arguments containing GPU ID and model details.
    :return: Normalized CLIP text features.
    """
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    model, _, _ = open_clip.create_model_and_transforms(
        model_name=args.clip_model,
        pretrained=args.pretrained,
        force_quick_gelu=True,
        device=device
    )
    model = model.to(device)
    model.eval()

    tokenizer = open_clip.get_tokenizer(args.clip_model)

    input_path = args.input_path
    input_path = input_path if input_path.endswith("/") else input_path + "/"
    scene_id = input_path.split("/")[-2]  # Get the scene ID from the input path
    print(f"Loading scene classes from {scene_id}...")
    scene_classes_path = os.path.join(input_path, f"scene_priorities_{scene_id}.txt")
    with open(scene_classes_path, 'r') as scene_classes_file:
        scene_classes_dict = json.load(scene_classes_file)
    print(f"Scene classes loaded: {len(scene_classes_dict)} classes")

    scene_classes = list(scene_classes_dict.keys())

    with torch.no_grad():
        zeroshot_weights = []
        for scene_class in scene_classes:
            texts = [template.format(scene_class) for template in prompt_templates] # format with class
            texts = tokenizer(texts).to(device)                         # tokenize
            class_embeddings = model.encode_text(texts).float()    # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)              # average multi templates
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        
        zeroshot_weights = torch.stack(zeroshot_weights, dim=0)
    print(f"Text features extracted with shape {zeroshot_weights.size()}!")
    
    del model
    torch.cuda.empty_cache()
    
    zeroshot_weights = zeroshot_weights.detach().cpu()  # Move to CPU for saving

    # save the extracted feature 'clip_features/'
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    print(f"Saving text features to {save_path}/scene_classes_features.pt")
    torch.save(zeroshot_weights, f'{save_path}/scene_classes_features.pt')


def get_clip_visual_features(images, device='cuda:0'):
    """
    Get CLIP visual features for given images.
    :param image: The input images.
    :param device: GPU device to use for computation.
    :return: Normalized CLIP visual features.
    """
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name="ViT-B-16",
        pretrained="dfn2b",
        force_quick_gelu=True,
        device=device
    )
    model = model.to(device)
    model.eval()

    # image = Image.open(image_path) (H, W, C) # .convert("RGB")
    # preprocess(image).unsqueeze(0).to(device)
    images = [preprocess(image) for image in images]  # Preprocess and add batch dimension
    images = torch.stack(images, dim=0).to(device)  # Stack images into a batch

    with torch.no_grad():
        image_features = model.encode_image(images).float()  # Get image features
        image_features /= image_features.norm(dim=-1, keepdim=True)

    print(f"Visual features extracted with shape {image_features.size()}!")

    del model
    torch.cuda.empty_cache()

    image_features = image_features.detach().cpu()  # Move to CPU for saving

    return image_features


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

# adopted in our i-SLNeRF paper
def extract_clip_features(args):
    """
    Extract CLIP pixel features from images in the specified directory.
    :param args: Arguments containing image path, model details, GPU ID, and save path.
    """
    # Set the device for computation
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=args.clip_model,
        pretrained=args.pretrained,
        force_quick_gelu=True,
        device=device
    )
    model.eval()

    image_paths = sorted(glob.glob(f'{args.input_path}/*'))

    for image_path in tqdm(image_paths):

        down_sample = args.downscale
        image_path = Path(image_path)
        image = Image.open(image_path).convert("RGB")  # Ensure the image is in RGB format
        image = image.resize((image.width // down_sample, image.height // down_sample),
                             resample=Resampling.BILINEAR)

        patch_sizes = [min(image.size)//5, min(image.size)//8, min(image.size)//10]

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
                    patch_chunk_feature = model.encode_image(patch_chunk).float()
                    for i in range(chunk_size):
                        patch_idx = chunk_idx*chunk_size + i
                        if patch_idx >= len(idxes): break

                        sum_feature[:, :, idxes[patch_idx][1]:idxes[patch_idx][3], idxes[patch_idx][0]:idxes[patch_idx][2]] += \
                            patch_chunk_feature[i:i+1, :, None, None]
                        count[:, :, idxes[patch_idx][1]:idxes[patch_idx][3], idxes[patch_idx][0]:idxes[patch_idx][2]] += 1

                image_feature.append(sum_feature / count)

        image_feature = torch.cat(image_feature).detach() # [scale, D, height, width]
        image_feature = image_feature.permute(0, 2, 3, 1).cpu()  # Change to [scale, height, width, D]
        # image_feature = image_feature.half()  # Reduce the feature size by half for memory efficiency
        print(f"Extracted features for {image_path.name} with shape {image_feature.size()}")

        # save the extracted feature 'clip_features/'
        save_path = args.save_path
        os.makedirs(save_path, exist_ok=True)
        print(f"Saving image features to {save_path}/{image_path.stem}.pt")
        torch.save(image_feature, f'{save_path}/{image_path.stem}.pt')


def save_clip_features_relevancy_map(clip_text_features, clip_vis_feature, save_name, args):
    """
    Save the relevancy map of CLIP features.
    :param clip_text_features: CLIP text features for scene classes [N2, D].
    :param clip_vis_feature: CLIP visual feature for an image [H, W, D].
    :param save_name: Name of the file to save the relevancy map.
    :param args: Arguments containing save path, gpu_id.
    """
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    H, W = clip_vis_feature.size(0), clip_vis_feature.size(1)  # Get height and width of the visual features

    clip_vis_feature = clip_vis_feature.reshape(-1, clip_vis_feature.size(-1)).to(device) # [N1, D], N1 is H*W, D is the feature dimension
    clip_text_features = clip_text_features.to(device)  # [N2, D], N2 is the number of scene classes
    clip_text_features_normalized = F.normalize(clip_text_features, dim=1) # [N2, D], N2 is the number of scene classes
    clip_vis_feature_normalized = F.normalize(clip_vis_feature, dim=1) # [N1, D], N1 is 1 or H*W, D is the feature dimension
    # Compute cosine similarity
    relevancy_map = torch.mm(clip_vis_feature_normalized, clip_text_features_normalized.T).float() # [N1,N2]

    # Save the relevancy map 'clip_relevancy_maps/'
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    print(f"Saving relevancy map to {save_path}/{save_name}.pt")
    torch.save(relevancy_map, f'{save_path}/{save_name}.pt')


def save_all_relevancy_maps_from_path(args):
    """
    Save all relevancy maps from the specified path.
    :param args: Arguments containing input path, save path, and GPU ID.
    """
    # Load CLIP text features
    clip_text_features = torch.load(os.path.join(args.input_path, "scene_classes_features.pt"))

    # Get all image feature files
    image_feature_files = sorted(glob.glob(f"{args.input_path}/*.pt"))
    print(f"Found {len(image_feature_files)} image feature files in {args.input_path}")

    for image_feature_file in tqdm(image_feature_files, desc="Processing images"):
        if Path(image_feature_file).stem == "scene_classes_features":
            continue  # Skip the text features file
        clip_image_feature = torch.load(image_feature_file, weights_only=True)  # Load the image feature
        save_name = Path(image_feature_file).stem  # Use the file name without extension as save name
        print(f"Processing {save_name} with shape {clip_image_feature.size()}")
        # Save the relevancy map
        save_clip_features_relevancy_map(clip_text_features, clip_image_feature, save_name, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract CLIP pixel features from images")
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--clip_model", type=str, default="ViT-B-16")
    parser.add_argument("--pretrained", type=str, default="dfn2b")
    parser.add_argument("--downscale", type=int, default=2, help="Downscale factor for image size")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--save_path", type=str)
    args = parser.parse_args()

    # extract_clip_features(args)
    # get_clip_text_features(args)
    # save_all_relevancy_maps_from_path(args)

