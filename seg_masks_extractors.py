# Author: Qian Liu
# Email: liu.qian.pro@gmail.com

import ast
import os
import numpy as np
import json
import argparse
import torch


def get_unique_labels_per_scene(input_path, output_path):
    """
    Extracts unique labels from the input file and writes them to the output file.
    
    Args:
        input_path (str): Path to the input file containing scene data.
        output_path (str): Path to the output file where unique labels will be saved.
    """
    unique_labels_per_view = []
    unique_labels = []
    for _, _, files in os.walk(input_path):
        for label_conf_file in files:
            if label_conf_file.endswith("label_conf.txt"):
                sub_path_list = label_conf_file.split("_")
                sub_path = f"{sub_path_list[0]}_{sub_path_list[1]}"
                label_conf_path = os.path.join(input_path, sub_path, label_conf_file)
                with open(label_conf_path, "r") as label_conf_file_obj:
                    label_confs = label_conf_file_obj.readlines()
                
                label_list = label_confs[0].split(":")[1].strip()
                label_list = ast.literal_eval(label_list)
                unique_labels_per_view.append(list(set(label_list)))
                unique_labels =[*unique_labels, *list(set(label_list))]
    
    scene_name = input_path.split("/")[-3]
    unique_labels = list(set(unique_labels))
    output_file_path = os.path.join(output_path, f"unique_labels_{scene_name}.txt")
    with open(output_file_path, "a") as output_file:
        output_file.write(f"all distinct labels for scene {scene_name}\n\n")
        output_file.write(f"per-view labels length: {len(unique_labels_per_view)}\n")
        output_file.write(f"per-view labels: {unique_labels_per_view}\n\n")
        output_file.write(f"total unique labels: {len(unique_labels)}\n")
        output_file.write(f"all labels: {unique_labels}")


def get_list_duplicates(input_list):
    """
    Computes statistics for a list of duplicates with indices.
    
    Args:
        input_list (list): List of values to compute duplicates and their indices.
    
    Returns:
        dict: {element: [indices]}
    """
    duplicates_dict = {num: [i for i, x in enumerate(input_list) if x == num]
                       for num in set(input_list) if input_list.count(num) >= 1}
    
    return duplicates_dict


def extract_label_conf_content(label_conf_path):
    """
    Extracts label confidence content from the input file.
    
    Args:
        label_conf_path (str): Path to the file containing label confidence data.
    
    Returns:
        Tuple[List, List]: A tuple containing a list of label names and a list of their corresponding confidence scores.
    """
    with open(label_conf_path, "r") as label_conf_file_obj:
        label_confs = label_conf_file_obj.readlines()
        
        # print(f"label_confs length: {len(label_confs)}")
        label_list = label_confs[0].split(":")[1].strip()
        label_list = ast.literal_eval(label_list)
        conf_list = [line.strip() for line in label_confs[1:]]
        conf_list_str = ""
        for line in conf_list:
            conf_list_str += line
        # print(f"conf_list_str: {conf_list_str}")
        conf_list = conf_list_str.split(":")[1].strip()[7:-1]
        conf_list = ast.literal_eval(conf_list)
    
    return label_list, conf_list
    

def sort_list(input_list, reverse=False):
    """
    Sorts a list of values.
    
    Args:
        input_list (list): List of values to be sorted.
        reverse (bool): If True, sorts the list in descending order. Default is False (ascending order).
    
    Returns:
        Tuple[List, List]: Sorted list of values and their corresponding indices.
    """
    sorted_indices = [i for i, _ in sorted(enumerate(input_list), key=lambda x: x[1],
                                           reverse=reverse)]
    sorted_input_list = sorted(input_list, reverse=reverse)

    return sorted_input_list, sorted_indices


def assign_merged_mask_values(traversing_mask, merged_mask, all_labels_dict,
                              label_list, conf_list, traversing_index, same_order=False,
                              traversed_index=0, view_path="", view_name=""):
    # instance_index = duplicate_dict[label_list[traversing_index]].index(traversing_index) + 1
    instance_index = traversing_index + 1
    print(f"Instance index: {instance_index}")
    if not same_order:
        merged_mask[traversing_mask == 1] = np.array([all_labels_dict[label_list[traversing_index]],
                                                     instance_index, 0.0])
        where_mask = np.where(traversing_mask == 1)
        indices_list = np.array([(row, col) for row, col in zip(where_mask[0], where_mask[1])])
        print(f"Indices list shape: {indices_list.shape}")
        for element in indices_list:
            merged_mask[element[0], element[1], 2] = conf_list[traversing_index] + \
                                (round(np.random.uniform(1.0, 10.01) * 0.001, ndigits=4) * \
                                np.random.choice([-1.0, 1.0]))
    else:
        print(f"traversed index: {traversed_index}")
        traversed_mask = np.loadtxt(os.path.join(view_path,
                                                 f"{view_name}_mask_{traversed_index}.npy"),
                                    encoding="utf-8")
        combined_mask = traversing_mask + traversed_mask
        # traversed_instance_index = duplicate_dict[label_list[traversed_index]].index(
        #                            traversed_index) + 1
        combined_instance_index = traversing_index + 1
        print(f"Combined instance index: {combined_instance_index}")
        # merged_mask[combined_mask == 2] = np.array([all_labels_dict[label_list[traversed_index]],
        #                                             traversed_instance_index, 0.0])
        
        # assign values in traversing_mask and traversed_mask overlapping elements
        where_combined_mask = np.where(combined_mask == 2)
        combined_indices_list = np.array([(row, col) for row, col in
                                         zip(where_combined_mask[0], where_combined_mask[1])])
        print(f"Combined indices list shape: {combined_indices_list.shape}")
        for element in combined_indices_list:
            if merged_mask[element[0], element[1], 2] < (conf_list[traversing_index] - 0.01001):
                confidence = conf_list[traversing_index] + \
                             (round(np.random.uniform(1.0, 10.01) * 0.001, ndigits=4) * \
                              np.random.choice([-1.0, 1.0]))
                merged_mask[element[0], element[1], 0] = all_labels_dict[label_list[traversing_index]]
                merged_mask[element[0], element[1], 1] = combined_instance_index
                merged_mask[element[0], element[1], 2] = confidence
        
        # assign values in traversing_mask but not in overlapping elements and not in traversed_mask
        where_remained_mask = np.where((traversing_mask == 1) & (combined_mask == 1))
        remained_indices_list = np.array([(row, col) for row, col in
                                         zip(where_remained_mask[0], where_remained_mask[1])])
        print(f"Remained indices list shape: {remained_indices_list.shape}")
        for element in remained_indices_list:
            merged_mask[element[0], element[1], 0] = all_labels_dict[label_list[traversing_index]]
            merged_mask[element[0], element[1], 1] = combined_instance_index
            confidence = conf_list[traversing_index] + \
                         (round(np.random.uniform(1.0, 10.01) * 0.001, ndigits=4) * \
                          np.random.choice([-1.0, 1.0]))
            merged_mask[element[0], element[1], 2] = confidence


def merge_one_view_masks(view_path, scene_priorities_dict, output_path):
    """
    Merges masks from a single view into a single mask file.
    
    Args:
        view_path (str): Path to the directory containing mask files for a single view.
        scene_priorities_dict (dict): Dictionary relates unique labels to their scene orders.
        output_path (str): Path to the output file where the merged mask will be saved.
    """
    view_name = os.path.basename(view_path)
    label_conf_path = os.path.join(view_path, f"{view_name}_label_conf.txt")
    label_list, conf_list = extract_label_conf_content(label_conf_path)
    print(f"Label list: {label_list}")
    print(f"Confidence list: {conf_list}")
    duplicate_dict = get_list_duplicates(label_list)
    print(f"Duplicate dictionary: {duplicate_dict}")
    all_labels = list(scene_priorities_dict.keys())
    print(f"All labels length: {len(all_labels)}")
    all_labels_dict = {label: (i + 1) for i, label in enumerate(all_labels)}
    label_orders = []
    for label in label_list:
        if label in scene_priorities_dict:
            label_orders.append(scene_priorities_dict[label])
        else:
            print(f"Label {label} not found in scene priorities dictionary.")
    if len(label_orders) != len(label_list):
        print(f"Warning: Label orders length {len(label_orders)} does not match label list length {len(label_list)}.")
        raise ValueError("Label orders and label list lengths do not match.")
    label_orders_sorted, label_orders_sorted_indices = sort_list(label_orders, reverse=True)
    print(f"Label orders: {label_orders}")
    print(f"Label orders sorted: {label_orders_sorted}")
    print(f"Label orders sorted indices: {label_orders_sorted_indices}")
    traversing_index = label_orders_sorted_indices[0]
    print(f"Traversing index: {traversing_index}")

    traversing_mask_file_path = os.path.join(view_path, f"{view_name}_mask_{traversing_index}.npy")
    if not os.path.exists(traversing_mask_file_path):
        print(f"Mask file {traversing_mask_file_path} does not exist. Skipping view {view_name}.")
        raise FileNotFoundError(f"Mask file {traversing_mask_file_path} does not exist.")
    traversing_mask = np.loadtxt(traversing_mask_file_path, encoding="utf-8")
    merged_mask = np.zeros_like(traversing_mask, shape=(*traversing_mask.shape, 3), dtype=np.float16)
    assign_merged_mask_values(traversing_mask, merged_mask,
                              all_labels_dict, label_list, conf_list, traversing_index)
    
    # traverse the sorted label orders and update the merged mask accordingly
    traversed_index = traversing_index
    for index in range(1, len(label_orders_sorted)):
        print(f"Current index: {index}")
        traversing_index = label_orders_sorted_indices[index]
        print(f"Traversing index: {traversing_index}")
        traversing_mask_file_path = os.path.join(view_path, f"{view_name}_mask_{traversing_index}.npy")
        if not os.path.exists(traversing_mask_file_path):
            print(f"Mask file {traversing_mask_file_path} does not exist. Skipping view {view_name}.")
            raise FileNotFoundError(f"Mask file {traversing_mask_file_path} does not exist.")
        traversing_mask = np.loadtxt(traversing_mask_file_path, encoding="utf-8")

        if label_orders[traversed_index] == label_orders[traversing_index]:
            print(f"Label {label_list[traversing_index]} has the same order as traversed label {label_list[traversed_index]}.")
            assign_merged_mask_values(traversing_mask, merged_mask,
                                      all_labels_dict, label_list, conf_list, traversing_index,
                                      same_order=True, traversed_index=traversed_index,
                                      view_path=view_path, view_name=view_name)
        else:
            print(f"Label {label_list[traversing_index]} has a different order than traversed label {label_list[traversed_index]}.")
            assign_merged_mask_values(traversing_mask, merged_mask,
                                      all_labels_dict, label_list, conf_list, traversing_index)
        traversed_index = traversing_index
    
    output_file_path = os.path.join(output_path, f"{view_name}.pt")
    merged_mask_tensor = torch.from_numpy(merged_mask)
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    print(f"Saving merged mask to {output_file_path}")
    torch.save(merged_mask_tensor, output_file_path)
        

def merge_all_views_masks(input_path, scene_priorities_dict_path, output_path):
    """
    Merges masks from all views into a single mask file.
    
    Args:
        input_path (str): Path to the directory containing mask files for all views.
        scene_priorities_dict_path (str): Path to the file containing scene priorities.
        output_path (str): Path to the output file where the merged mask will be saved.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    
    with open(scene_priorities_dict_path, "r") as spdp_file:
        scene_priorities_dict = json.load(spdp_file)
    print(f"Scene priorities dictionary loaded from {scene_priorities_dict_path}")

    for _, view_paths, _ in os.walk(input_path):
        for view_path_name in view_paths:
            view_path = os.path.join(input_path, view_path_name)
            print(f"Processing view path: {view_path}")
            merge_one_view_masks(view_path, scene_priorities_dict, output_path)
            print(f"Merged masks saved to {output_path}")


def rename_merged_masks(input_path, output_path):
    """
    Renames merged mask files in the input directory and saves them to the output directory.
    
    Args:
        input_path (str): Path to the directory containing merged mask files.
        output_path (str): Path to the directory where renamed files will be saved.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    
    for _, _, files in os.walk(input_path):
        for file_name in files:
            if file_name.endswith(".pt"):
                new_file_name = file_name.replace("_merged_mask", "")
                new_file_path = os.path.join(output_path, new_file_name)
                old_file_path = os.path.join(input_path, file_name)
                print(f"Renaming {old_file_path} to {new_file_path}")
                old_mask_tensor = torch.load(old_file_path)
                torch.save(old_mask_tensor, new_file_path)


def save_same_diff_two_unique_labels_dicts(unique_labels_per_scene, unique_labels_all, output_path):
    """
    Computes the difference between two unique labels dictionaries.
    
    Args:
        unique_labels_per_scene (list): List containing unique labels for a specific scene.
        unique_labels_all (dict): Dictionary containing all unique labels.
        output_path (str): Path to the output file where different labels will be saved.
    """
    same_labels, different_labels = [], []
    for label in unique_labels_per_scene:
        if label not in unique_labels_all.keys():
            different_labels.append(label)
        else:
            same_labels.append(label)

    scene_id = output_path.split("/")[-2]
    output_same_file_path = os.path.join(output_path, f"same_labels_{scene_id}.txt")
    with open(output_same_file_path, "w") as output_same_file:
        json.dump({len(same_labels):same_labels}, output_same_file, indent=4)
    print(f"same labels length: {len(same_labels)}")
    output_diff_file_path = os.path.join(output_path, f"different_labels_{scene_id}.txt")
    with open(output_diff_file_path, "w") as output_diff_file:
        json.dump({len(different_labels):different_labels}, output_diff_file, indent=4)
    print(f"Different labels length: {len(different_labels)}")

    output_scene_file_path = os.path.join(output_path, f"scene_labels_{scene_id}.txt")
    with open(output_scene_file_path, "w") as output_scene_file:
        json.dump(unique_labels_per_scene, output_scene_file)
    print(f"All scene labels length: {len(unique_labels_per_scene)}")

    print(f"Same labels saved to {output_same_file_path}")
    print(f"Different labels saved to {output_diff_file_path}")
    print(f"All scene labels saved to {output_scene_file_path}")


def extract_labels_from_scene_labels_file(scene_file_path):
    """
    Extracts unique labels from a scene labels file.
    
    Args:
        scene_file_path (str): Path to the scene labels file.
    
    Returns:
        list: A list of unique labels extracted from the scene file.
    """
    with open(scene_file_path, "r") as scene_file:
        scene_file_content = scene_file.readlines()
    
    for line in scene_file_content:
        if line.startswith("all labels"):
            unique_labels = line.split(":")[1].strip()
            unique_labels = ast.literal_eval(unique_labels)
            break
    
    return unique_labels


def load_save_same_diff_unique_labels(input_path, output_path):
    """
    Loads unique labels from a file and saves the same and different labels with another file.
    
    Args:
        input_path (str): Path to the input file containing unique labels for a scene.
        output_path (str): Path to the output file where different labels will be saved.
    """
    scene_id = input_path.split("/")[-2]
    unique_scene_file = os.path.join(input_path, f"unique_labels_{scene_id}.txt")
    with open(os.path.join(input_path, "unique_labels_ordered.txt"), "r") as unique_labels_file:
        unique_labels_all = json.load(unique_labels_file)
    unique_labels_per_scene = extract_labels_from_scene_labels_file(unique_scene_file)

    save_same_diff_two_unique_labels_dicts(unique_labels_per_scene, unique_labels_all, output_path)


def save_priorities_for_a_scene(unique_labels_per_scene, unique_labels_all, output_path):
    """
    Computes priorities for a scene based on unique labels.
    
    Args:
        unique_labels_per_scene (list): List containing unique labels for a specific scene.
        unique_labels_all (dict): Dictionary containing all unique labels and their priorities.
        output_path (str): Path to the output file where dictionary of scene priorities will be saved.
    """
    priorities = {}
    for label in unique_labels_all.keys():
        for scene_label in unique_labels_per_scene:
            if scene_label in label or label in scene_label:
                priorities[scene_label] = unique_labels_all[label]
    
    scene_id = output_path.split("/")[-2]
    output_file_path = os.path.join(output_path, f"scene_priorities_{scene_id}.txt")
    with open(output_file_path, "w") as output_file:
        json.dump(priorities, output_file, indent=4)
    
    print(f"Scene priorities saved to {output_file_path}")


def load_save_priorities(input_path, output_path):
    """
    Loads unique labels from a file and saves the priorities for a scene.
    
    Args:
        input_path (str): Path to the input file containing scene unique labels and all labels for a scene.
        output_path (str): Path to the output file where priorities will be saved.
    """
    scene_id = input_path.split("/")[-2]
    unique_scene_file = os.path.join(input_path, f"unique_labels_{scene_id}.txt")
    with open(os.path.join(input_path, "unique_labels_ordered.txt"), "r") as unique_labels_file:
        unique_labels_all = json.load(unique_labels_file)
    unique_labels_per_scene = extract_labels_from_scene_labels_file(unique_scene_file)

    save_priorities_for_a_scene(unique_labels_per_scene, unique_labels_all, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract instance segmentation & confidence masks from images")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input images directory")
    parser.add_argument("--scene_id", type=int, default=0, help="Scene ID for the input images")
    parser.add_argument("--save_path", type=str, default="", help="Path to save output masks")
    args = parser.parse_args()

    # get_unique_labels_per_scene(args.input_path, args.save_path)
    # print(f"Unique labels extracted and saved to {args.save_path}")
    # load_save_same_diff_unique_labels(args.input_path, args.save_path)
    # ordered_unique_labels_dict_path = os.path.join(args.input_path,
    #                                                "unique_labels_ordered.txt")
    # rename_merged_masks(args.input_path, args.output_path)
    # merge_all_views_masks(input_path, ordered_unique_labels_dict_path, output_path)

    print("All Processing complete.")
    