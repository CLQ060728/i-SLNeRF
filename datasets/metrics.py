# Author: Qian Liu
# Email: liu.qian.pro@gmail.com

from typing import List, Union

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from torch import Tensor
from torch.nn import functional as F
from tqdm import tqdm
from radiance_fields import RadianceField


def normalize_depth(depth: Tensor, max_depth: float = 1000.0):
    return torch.clamp(depth / max_depth, 0.0, 1.0)


def compute_valid_depth_rmse(prediction: Tensor, target: Tensor, max_depth: float) -> float:
    """
    Computes the root mean squared error (RMSE) between the predicted and target depth values,
    only considering the valid rays (where target > 0).

    Args:
    - prediction (Tensor): predicted depth values
    - target (Tensor): target depth values

    Returns:
    - float: RMSE between the predicted and target depth values, only considering the valid rays
    """
    prediction, target = prediction.squeeze(), target.squeeze()
    # valid_mask = target > 0
    valid_mask = (target > 0.0) & (target <= max_depth)
    prediction = normalize_depth(prediction[valid_mask], max_depth=max_depth)
    target = normalize_depth(target[valid_mask], max_depth=max_depth)
    # prediction = prediction[valid_mask]
    # target = target[valid_mask]
    return F.mse_loss(prediction, target).sqrt().item()


def compute_psnr(prediction: Tensor, target: Tensor) -> float:
    """
    Computes the Peak Signal-to-Noise Ratio (PSNR) between the prediction and target tensors.

    Args:
        prediction (torch.Tensor): The predicted tensor.
        target (torch.Tensor): The target tensor.

    Returns:
        float: The PSNR value between the prediction and target tensors.
    """
    if not isinstance(prediction, Tensor):
        prediction = Tensor(prediction)
    if not isinstance(target, Tensor):
        target = Tensor(target).to(prediction.device)
    return (-10 * torch.log10(F.mse_loss(prediction, target))).item()


def compute_ssim(
    prediction: Union[Tensor, np.ndarray], target: Union[Tensor, np.ndarray]
) -> float:
    """
    Computes the Structural Similarity Index (SSIM) between the prediction and target images.

    Args:
        prediction (Union[Tensor, np.ndarray]): The predicted image.
        target (Union[Tensor, np.ndarray]): The target image.

    Returns:
        float: The SSIM value between the prediction and target images.
    """
    if isinstance(prediction, Tensor):
        prediction = prediction.cpu().numpy()
    if isinstance(target, Tensor):
        target = target.cpu().numpy()
    assert target.max() <= 1.0 and target.min() >= 0.0, "target must be in range [0, 1]"
    assert (
        prediction.max() <= 1.0 and prediction.min() >= 0.0
    ), "prediction must be in range [0, 1]"
    return ssim(target, prediction, data_range=1.0, channel_axis=-1)


def compute_scene_flow_metrics(pred: Tensor, labels: Tensor):
    """
    Computes the scene flow metrics between the predicted and target scene flow values.
    # modified from https://github.com/Lilac-Lee/Neural_Scene_Flow_Prior/blob/0e4f403c73cb3fcd5503294a7c461926a4cdd1ad/utils.py#L12

    Args:
        pred (Tensor): predicted scene flow values
        labels (Tensor): target scene flow values
    Returns:
        dict: scene flow metrics
    """
    l2_norm = torch.sqrt(
        torch.sum((pred - labels) ** 2, 2)
    ).cpu()  # Absolute distance error.
    labels_norm = torch.sqrt(torch.sum(labels * labels, 2)).cpu()
    relative_err = l2_norm / (labels_norm + 1e-20)

    EPE3D = torch.mean(l2_norm).item()  # Mean absolute distance error

    # NOTE: Acc_5
    error_lt_5 = torch.BoolTensor((l2_norm < 0.05))
    relative_err_lt_5 = torch.BoolTensor((relative_err < 0.05))
    acc3d_strict = torch.mean((error_lt_5 | relative_err_lt_5).float()).item()

    # NOTE: Acc_10
    error_lt_10 = torch.BoolTensor((l2_norm < 0.1))
    relative_err_lt_10 = torch.BoolTensor((relative_err < 0.1))
    acc3d_relax = torch.mean((error_lt_10 | relative_err_lt_10).float()).item()

    # NOTE: outliers
    l2_norm_gt_3 = torch.BoolTensor(l2_norm > 0.3)
    relative_err_gt_10 = torch.BoolTensor(relative_err > 0.1)
    outlier = torch.mean((l2_norm_gt_3 | relative_err_gt_10).float()).item()

    # NOTE: angle error
    unit_label = labels / (labels.norm(dim=-1, keepdim=True) + 1e-7)
    unit_pred = pred / (pred.norm(dim=-1, keepdim=True) + 1e-7)

    # it doesn't make sense to compute angle error on zero vectors
    # we use a threshold of 0.1 to avoid noisy gt flow
    non_zero_flow_mask = labels_norm > 0.1
    unit_label = unit_label[non_zero_flow_mask]
    unit_pred = unit_pred[non_zero_flow_mask]

    eps = 1e-7
    dot_product = (unit_label * unit_pred).sum(-1).clamp(min=-1 + eps, max=1 - eps)
    dot_product[dot_product != dot_product] = 0  # Remove NaNs
    angle_error = torch.acos(dot_product).mean().item()

    return {
        "EPE3D": EPE3D,
        "acc3d_strict": acc3d_strict,
        "acc3d_relax": acc3d_relax,
        "outlier": outlier,
        "angle_error": angle_error,
    }


def knn_predict(
    queries: Tensor,
    memory_bank: Tensor,
    memory_labels: Tensor,
    n_classes: int,
    knn_k: int = 1,
    knn_t: float = 0.1,
) -> Tensor:
    """
    Compute kNN predictions for each query sample in memory_bank based on memory_labels.

    Args:
        queries (Tensor): query feature vectors
        memory_bank (Tensor): memory feature vectors
        memory_labels (Tensor): memory labels
        n_classes (int): number of classes
        knn_k (int, optional): number of nearest neighbors. Defaults to 1.
        knn_t (float, optional): temperature for softmax. Defaults to 0.1.

    Returns:
        Tensor: kNN predictions for each query sample in queries based on memory_bank and memory_labels
    """
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(queries, memory_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(
        memory_labels.expand(queries.size(0), -1), dim=-1, index=sim_indices
    )
    sim_weight = (sim_weight / knn_t).exp()
    # counts for each class
    one_hot_label = torch.zeros(
        queries.size(0) * knn_k, n_classes, device=sim_labels.device
    )
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(
        dim=-1, index=sim_labels.view(-1, 1), value=1.0
    )
    # weighted score ---> [B, C]
    pred_scores = torch.sum(
        one_hot_label.view(queries.size(0), -1, n_classes)
        * sim_weight.unsqueeze(dim=-1),
        dim=1,
    )
    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels


def knn_predict(
    queries: Tensor,
    memory_bank: Tensor,
    memory_labels: Tensor,
    n_classes: int,
    knn_k: int = 1,
    knn_t: float = 0.1,
    similarity: str = "cosine",
) -> Tensor:
    """
    Compute kNN predictions for each query sample in memory_bank based on memory_labels.

    Args:
        queries (Tensor): query feature vectors [N_q, D]
        memory_bank (Tensor): Transposed memory feature vectors: [D, N_m]
        memory_labels (Tensor): memory labels
        n_classes (int): number of classes
        knn_k (int, optional): number of nearest neighbors. Defaults to 1.
        knn_t (float, optional): temperature for softmax. Defaults to 0.1.
        similarity (str, optional): similarity metric to use. Defaults to "cosine".

    Returns:
        Tensor: kNN predictions for each query sample in queries based on memory_bank and memory_labels
    """
    if similarity == "cosine":
        # compute cos similarity between each feature vector and feature bank ---> [N_q, N_m]
        assert queries.size(-1) == memory_bank.size(0)
        memory_bank = memory_bank.T
        memory_bank = memory_bank / (memory_bank.norm(dim=-1, keepdim=True) + 1e-7)
        similarity_matrix = torch.mm(
            queries / (queries.norm(dim=-1, keepdim=True) + 1e-7),
            memory_bank.T,
        )
    elif similarity == "l2":
        # compute the L2 distance using broadcasting
        queries_expanded = queries.unsqueeze(1)  # Shape becomes [N_q, 1, D]
        memory_bank_expanded = memory_bank.T.unsqueeze(0)  # Shape becomes [1, N_m, D]
        dist_matrix = torch.norm(
            queries_expanded - memory_bank_expanded, dim=2
        )  # Shape becomes [N_q, N_m]
        # Invert the distances to get the similarity
        similarity_matrix = 1 / (dist_matrix + 1e-9)  # Shape remains [N_q, N_m]
    else:
        raise ValueError(f"similarity {similarity} is not supported")
    # [N_q, K]
    sim_weight, sim_indices = similarity_matrix.topk(k=knn_k, dim=-1)
    sim_labels = torch.gather(
        memory_labels.expand(queries.size(0), -1), dim=-1, index=sim_indices
    )
    # scale by temperature
    sim_weight = (sim_weight / knn_t).exp()
    # counts for each class
    one_hot_label = torch.zeros(
        queries.size(0) * knn_k, n_classes, device=sim_labels.device
    )
    # [N_q * K, num_class]
    one_hot_label = one_hot_label.scatter(
        dim=-1, index=sim_labels.view(-1, 1), value=1.0
    )
    # [N_q, num_class]
    pred_scores = torch.sum(
        one_hot_label.view(queries.size(0), -1, n_classes)
        * sim_weight.unsqueeze(dim=-1),
        dim=1,
    )
    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels

