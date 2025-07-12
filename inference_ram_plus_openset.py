'''
 * The Recognize Anything Plus Model (RAM++) inference on unseen classes
 * Written by Xinyu Huang
'''
import argparse
import numpy as np
import random

import torch

from PIL import Image
from ram.models import ram_plus
from ram import inference_ram_openset as inference
from ram import get_transform

from ram.utils import build_openset_llm_label_embedding
from torch import nn
import json
import os

parser = argparse.ArgumentParser(
    description='Tag2Text inferece for tagging and captioning')
parser.add_argument('--image',
                    metavar='DIR',
                    help='path to dataset',
                    default='images/openset_example.jpg')
parser.add_argument('--pretrained',
                    metavar='DIR',
                    help='path to pretrained model',
                    default='pretrained/ram_plus_swin_large_14m.pth')
parser.add_argument('--image-size',
                    default=384,
                    type=int,
                    metavar='N',
                    help='input image size (default: 384)')
parser.add_argument('--llm_tag_des',
                    metavar='DIR',
                    help='path to LLM tag descriptions',
                    default='datasets/openimages_rare_200/openimages_rare_200_llm_tag_descriptions.json')
parser.add_argument('--output',
                    metavar='DIR',
                    help='path to output tags',
                    default='./output/image_tags.txt')

if __name__ == "__main__":

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = get_transform(image_size=args.image_size)

    #######load model
    model = ram_plus(pretrained=args.pretrained,
                             image_size=args.image_size,
                             vit='swin_l')

    #######set openset interference

    print('Building tag embedding:')
    with open(args.llm_tag_des, 'rb') as fo:
        llm_tag_des = json.load(fo)
    openset_label_embedding, openset_categories = build_openset_llm_label_embedding(llm_tag_des)

    model.tag_list = np.array(openset_categories)
    
    model.label_embed = nn.Parameter(openset_label_embedding.float())

    model.num_class = len(openset_categories)
    # the threshold for unseen categories is often lower
    model.class_threshold = torch.ones(model.num_class) * 0.5
    #######

    model.eval()

    model = model.to(device)
    if os.path.isdir(args.image):
        with open(args.output, 'a') as out:
            for _, _, files in os.walk(args.image):
                for file in files:
                    if file.endswith('.jpg') or file.endswith('.png'):
                        image_path = os.path.join(args.image, file)
                        print("Processing: ", image_path)
                        image = transform(Image.open(image_path)).unsqueeze(0).to(device)

                        res = inference(image, model)
                        print("Image Tags: ", res)
                        out.write(file + ': ' + str(res) + '\n')
    else:
        print("Processing: ", args.image)
        image = transform(Image.open(args.image)).unsqueeze(0).to(device)

        res = inference(image, model)
        print("Image Tags: ", res)
        with open(args.output, 'a') as out:
            out.write(args.image + ': ' + str(res) + '\n')

