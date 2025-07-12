# Author: Qian Liu
# Email: liu.qian.pro@gmail.com

import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
import argparse
import multiprocessing as mp
import concurrent.futures as CF


parser = argparse.ArgumentParser(
    description='Grounded SAM2 inference in batch mode')
parser.add_argument('--prompt_path',
                    default="",
                    help='text prompt path',
                    type=str)
parser.add_argument('--img_path',
                    default="",
                    help='input image path',
                    type=str)
parser.add_argument('--sam2_checkpoint',
                    default="./checkpoints/sam2.1_hiera_large.pt",
                    help='path to sam2 checkpoint',
                    type=str)
parser.add_argument('--sam2_model_config',
                    default="configs/sam2.1/sam2.1_hiera_l.yaml",
                    help='path to sam2 model config',
                    type=str)
parser.add_argument('--grounding_dino_config',
                    default="grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                    help='path to grounding dino config',
                    type=str)
parser.add_argument('--grounding_dino_checkpoint',
                    default="gdino_checkpoints/groundingdino_swint_ogc.pth",
                    help='path to grounding dino checkpoint',
                    type=str)
parser.add_argument('--num_workers',
                    default=2,
                    type=int,
                    help='number of workers for Grounded SAM2 inference')
parser.add_argument('--box_threshold',
                    default=0.30,
                    type=float,
                    help='box threshold for grounding dino')
parser.add_argument('--text_threshold',
                    default=0.30,
                    type=float,
                    help='text threshold for grounding dino')
parser.add_argument('--output_dir',
                    default="outputs",
                    type=str,
                    help='output directory')
parser.add_argument('--batch_lower_bound',
                    default=1,
                    type=int,
                    help='batch lower bound for tasks')
parser.add_argument('--batch_upper_bound',
                    default=100,
                    type=int,
                    help='batch upper bound for tasks')
parser.add_argument('--dump_json_results',
                    action='store_true',
                    help='dump json results')


def inference_on_single_img(args, sam2_checkpoint, model_cfg, img_path, text,
                            output_path, device):
    """
    Get mask related artefacts
    :return: mask related artefacts
    """
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    # build grounding dino model
    grounding_model = load_model(
        model_config_path=args.grounding_dino_config,
        model_checkpoint_path=args.grounding_dino_checkpoint,
        device=device
    )
    
    image_source, image = load_image(img_path)

    sam2_predictor.set_image(image_source)

    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image,
        caption=text,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold
    )
    print(f"labels shape: {len(labels)}")
    print(f"labels: {labels}")
    print(f"confidences shape: {len(confidences)}")
    print(f"confidences: {confidences}")
    file_name = os.path.basename(img_path)[:-4]
    label_conf_file = os.path.join(output_path, f"{file_name}_label_conf.txt")
    with open(label_conf_file, "w") as lcf:
        lcf.write(f"labels: {labels}\n")
        lcf.write(f"confidences: {confidences}\n")

    # process the box prompt for SAM 2
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()


    # FIXME: figure how does this influence the G-DINO model
    torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    masks, scores, _ = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    """
    Post-process the output of the model to get the masks, scores, and logits for visualization
    """
    # convert the shape to (n, H, W)
    if masks.ndim == 4:
        masks = masks.squeeze(1)


    confidences = confidences.numpy().tolist()
    class_names = labels

    class_ids = np.array(list(range(len(class_names))))

    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence
        in zip(class_names, confidences)
    ]
    for mask_id in range(len(masks)):
        # np.savetxt(os.path.join(output_path, f"{file_name}_mask_{mask_id}.npy"), masks[mask_id])
        np.save(os.path.join(output_path, f"{file_name}_mask_{mask_id}.npy"), masks[mask_id])
    """
    Visualize image with supervision useful API
    """
    img = cv2.imread(img_path)
    detections = sv.Detections(
        xyxy=input_boxes,  # (n, 4)
        mask=masks.astype(bool),  # (n, h, w)
        class_id=class_ids
    )

    # box_annotator = sv.BoxAnnotator()
    # annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

    # label_annotator = sv.LabelAnnotator()
    # annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    # file_name = os.path.basename(img_path)[:-4]
    # cv2.imwrite(os.path.join(output_path, f"{file_name}_bbox.jpg"), annotated_frame)

    mask_annotator = sv.MaskAnnotator()
    annotated_frame = mask_annotator.annotate(scene=img.copy(), detections=detections)
    # annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    cv2.imwrite(os.path.join(output_path, f"{file_name}_mask.jpg"), annotated_frame)

    """
    Dump the results in standard format and save as json files
    """

    def single_mask_to_rle(mask):
        rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
        rle["counts"] = rle["counts"].decode("utf-8")
        return rle

    if args.dump_json_results:
        # convert mask into rle format
        mask_rles = [single_mask_to_rle(mask) for mask in masks]
        input_boxes = input_boxes.tolist()
        scores = scores.tolist()
        # save the results in standard format
        results = {
            "image_path": img_path,
            "annotations" : [
                {
                    "class_name": class_name,
                    "bbox": box,
                    "segmentation": mask_rle,
                    "score": score,
                }
                for class_name, box, mask_rle, score in zip(class_names, input_boxes, mask_rles, scores)
            ],
            "box_format": "xyxy",
            "img_width": w,
            "img_height": h,
        }
        
        with open(os.path.join(output_path, f"{file_name}.json"), "w") as f:
            json.dump(results, f, indent=4)
    
    del sam2_model
    del sam2_predictor
    del grounding_model
    del image_source
    del image
    del masks
    del boxes
    del scores
    del input_boxes
    del confidences
    torch.cuda.empty_cache()


if __name__ == "__main__":
    args = parser.parse_args()

    # environment settings
    # use bfloat16
    # build SAM2 image predictor
    sam2_checkpoint = args.sam2_checkpoint
    model_cfg = args.sam2_model_config

    # setup the input image and text prompt for SAM 2 and Grounding DINO
    # VERY important: text queries need to be lowercased + end with a dot

    with open(args.prompt_path, "r") as prompt_file:
        prompts = prompt_file.readlines()

    prompts_dict = {}
    for prompt_line in prompts:
        prompt_line = prompt_line.strip()
        prompt_line_split = prompt_line.split(":")
        if len(prompt_line_split) == 2:
            prompt_line_split[0] = prompt_line_split[0].strip()
            prompt_line_split[1] = prompt_line_split[1].strip()
            prompts_dict[prompt_line_split[0]] = prompt_line_split[1]
        else:
            print(f"Invalid line in prompt file: {prompt_line}")
            continue
    # print(f"prompts_dict: {prompts_dict}")
    prompt_root = "/"
    for part in args.prompt_path.split("/")[:-1]:
        prompt_root = os.path.join(prompt_root, part)
    print(f"Prompt root: {prompt_root}")

    if os.path.isdir(args.img_path):
        output_subdir = args.img_path.split("/")[-3]
        output_path_root = os.path.join(args.output_dir, output_subdir)
        print(f"Output path root: {output_path_root}")
        if not os.path.exists(os.path.join(prompt_root, "file_names.txt")):
            file_names_list = []
            for _, _, files in os.walk(args.img_path):
                    for file in files:
                        if file.endswith(('.jpg', '.png', '.jpeg')):
                            file_names_list.append(file)
            with open(os.path.join(prompt_root, "file_names.txt"), "w") as file_names:
                json.dump(file_names_list, file_names)
        else:
            with open(os.path.join(prompt_root, "file_names.txt"), "r") as file_names:
                file_names_list = json.load(file_names)
        
        taskFutures = list()
        context = mp.get_context("spawn")
        num_workers = args.num_workers
        with CF.ProcessPoolExecutor(max_workers=num_workers, mp_context=context) as executor:
            for idx in range(args.batch_lower_bound, args.batch_upper_bound, 1):
                img_path = os.path.join(args.img_path, file_names_list[idx])
                text = prompts_dict[file_names_list[idx]] + " |"
                text = text.replace("|", ".")
                print(f"Processing image: {file_names_list[idx]}")
                print(f"Text prompt: {text}")
                output_path = os.path.join(output_path_root, file_names_list[idx][:-4])
                os.makedirs(output_path, exist_ok=True)
                print(f"Current output path: {output_path}")
                device = f"cuda:{idx % num_workers}" if torch.cuda.is_available() else "cpu"
                print(f"Using device: {device}")
                if idx == 0 or (idx != 0 and idx % num_workers > 0):
                    taskFutures.append(executor.submit(inference_on_single_img, args,
                                                        sam2_checkpoint, model_cfg, 
                                                        img_path, text, output_path,
                                                        device))
                elif idx != 0 and idx % num_workers == 0:
                    done, not_done = CF.wait(taskFutures, return_when=CF.ALL_COMPLETED)
                    for future in done:
                        print(future)
                    for futurenot in not_done:
                        print(futurenot)
                    taskFutures.clear()
                    taskFutures.append(executor.submit(inference_on_single_img, args,
                                                        sam2_checkpoint, model_cfg, 
                                                        img_path, text, output_path,
                                                        device))
                
            done, not_done = CF.wait(taskFutures, return_when=CF.ALL_COMPLETED)
            for future in done:
                print(future)
            for futurenot in not_done:
                print(futurenot)
            taskFutures.clear()
    else:
        output_path = args.output_dir
        text = prompts_dict[os.path.basename(args.img_path)]
        text = text + " |"
        text = text.replace("|", ".")
        print(f"Processing image: {os.path.basename(args.img_path)}")
        print(f"Text prompt: {text}")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        inference_on_single_img(args, sam2_checkpoint, model_cfg, args.img_path,
                                text, output_path, device)
    