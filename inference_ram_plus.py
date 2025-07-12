'''
 * The Recognize Anything Plus Model (RAM++)
 * Written by Xinyu Huang
'''
import argparse
import numpy as np
import random
import os 
import torch

from PIL import Image
from ram.models import ram_plus
from ram import inference_ram as inference
from ram import get_transform


parser = argparse.ArgumentParser(
    description='Tag2Text inferece for tagging and captioning')
parser.add_argument('--image',
                    metavar='DIR',
                    help='path to dataset',
                    default='images/demo/demo1.jpg')
parser.add_argument('--pretrained',
                    metavar='DIR',
                    help='path to pretrained model',
                    default='pretrained/ram_plus_swin_large_14m.pth')
parser.add_argument('--image-size',
                    default=384,
                    type=int,
                    metavar='N',
                    help='input image size (default: 448)')
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
    model.eval()

    model = model.to(device)

    # image = transform(Image.open(args.image)).unsqueeze(0).to(device)

    # res = inference(image, model)
    # print("Image Tags: ", res[0])
    # print("图像标签: ", res[1])

    if os.path.isdir(args.image):
        with open(args.output, 'a') as out:
            for _, _, files in os.walk(args.image):
                for file in files:
                    if file.endswith('.jpg') or file.endswith('.png'):
                        image_path = os.path.join(args.image, file)
                        print("Processing: ", image_path)
                        image = transform(Image.open(image_path)).unsqueeze(0).to(device)

                        res = inference(image, model)
                        print("Image Tags: ", res[0])
                        out.write(file + ': ' + str(res[0]) + '\n')
    else:
        print("Processing: ", args.image)
        image = transform(Image.open(args.image)).unsqueeze(0).to(device)

        res = inference(image, model)
        print("Image Tags: ", res[0])
        with open(args.output, 'a') as out:
            out.write(args.image + ': ' + str(res[0]) + '\n')