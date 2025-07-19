# Author: Qian Liu
# Email: liu.qian.pro@gmail.com

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from nerfacc import accumulate_along_rays
from torch import Tensor


def normalize_depth(depth: Tensor, max_depth: float = 1000.0):
    return torch.clamp(depth / max_depth, 0.0, 1.0)


class Loss(nn.Module):
    """
    Base class for defining custom loss functions.

    Args:
        coef (float): Coefficient to scale the loss by.
        check_nan (bool): Whether to check if the loss is NaN.
        reduction (str): Type of reduction to apply to the loss. Can be "mean" or "none".

    Methods:
        __call__(self, *args, name: str, **kwargs): Computes the loss.
        set_coef(self, coef: float): Sets the coefficient to scale the loss by.
        return_loss(self, name: str, loss: Tensor): Returns the loss scaled by the coefficient.
    """

    def __init__(self, coef: float = 1.0, check_nan: bool = False, reduction="mean"):
        super(Loss, self).__init__()
        self.coef = coef
        self.check_nan = check_nan
        assert reduction in ["mean", "none"]
        self.reduction = reduction

    def __call__(self, *args, name: str, **kwargs):
        """
        Computes the loss.

        Args:
            *args: Variable length argument list.
            name (str): Name of the loss.
            **kwargs: Arbitrary keyword arguments.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError()

    def set_coef(self, coef: float):
        """
        Sets the coefficient to scale the loss by.

        Args:
            coef (float): Coefficient to scale the loss by.
        """
        self.coef = coef

    def return_loss(self, name: str, loss: Tensor):
        """
        Returns the loss scaled by the coefficient.

        Args:
            name (str): Name of the loss.
            loss (Tensor): Loss tensor.

        Returns:
            dict: Dictionary containing the scaled loss.
        """
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "none":
            loss = loss
        else:
            raise NotImplementedError()
        if self.check_nan:
            if torch.isnan(loss):
                raise ValueError(f"Loss {name} is NaN.")
        return {name: loss * self.coef}


class RealValueLoss(Loss):
    """
    A class representing a real value loss function.

    Args:
        loss_type (Literal["l1", "l2", "smooth_l1"]): The type of loss function to use.
        coef (float): The coefficient to multiply the loss by.
        name (str): The name of the loss function.
        reduction (str): The reduction method to use.
        check_nan (bool): Whether to check for NaN values in the loss.

    Attributes:
        loss_type (Literal["l1", "l2", "smooth_l1"]): The type of loss function being used.
        loss_fn (function): The loss function being used.
        name (str): The name of the loss function.
    """

    def __init__(
        self,
        loss_type: Literal["l1", "l2", "smooth_l1"] = "l2",
        coef: float = 1.0,
        name="rgb",
        reduction="mean",
        check_nan=False,
    ):
        super(RealValueLoss, self).__init__(coef, check_nan, reduction)
        self.loss_type = loss_type
        if self.loss_type == "l1":
            self.loss_fn = F.l1_loss
        elif self.loss_type == "l2":
            self.loss_fn = F.mse_loss
        elif self.loss_type == "smooth_l1":
            self.loss_fn = F.smooth_l1_loss
        else:
            raise NotImplementedError(f"Unknown loss type: {loss_type}")
        self.name = f"{name}_loss_{self.loss_type}"

    def __call__(
        self,
        predicted: Tensor,
        gt: Tensor,
        mask: Tensor = None,
        name: str = None,
        coef: float = 1.0,
    ):
        """
        Compute the loss between the predicted and ground truth values.

        Args:
            predicted (Tensor): The predicted values.
            gt (Tensor): The ground truth values.
            mask (Tensor): The mask to apply to the loss.
            name (str): The name of the loss function.
            coef (float): The coefficient to multiply the loss by.

        Returns:
            The loss value.
        """
        gt, predicted = gt.squeeze(), predicted.squeeze()
        loss = self.loss_fn(predicted, gt, reduction="none")
        if mask is not None:
            loss = loss * mask.squeeze()
        name = self.name if name is None else name
        return self.return_loss(name, loss * coef)


class SkyLoss(Loss):
    def __init__(
        self,
        loss_type: Literal["weights_based", "opacity_based"] = "weights_based",
        coef: float = 0.01,
        reduction="mean",
        check_nan=False,
    ):
        super(SkyLoss, self).__init__(coef, check_nan, reduction)
        self.loss_type = loss_type
        if self.loss_type == "weights_based":
            self.loss_fn = self._reduce_weights_towards_zero
        elif self.loss_type == "opacity_based":
            self.loss_fn = self._binary_entropy_loss
        else:
            raise NotImplementedError(f"Unknown loss type: {loss_type}")
        self.name = f"sky_loss_{self.loss_type}"

    def _reduce_weights_towards_zero(self, weights: Tensor, sky_mask: Tensor):
        sky_loss = (weights.square().sum(-1) * sky_mask).mean()
        return sky_loss

    def _binary_entropy_loss(self, opacity: Tensor, sky_mask: Tensor):
        sky_loss = F.binary_cross_entropy(
            opacity.squeeze(), 1 - sky_mask.float(), reduction="none"
        )
        return sky_loss

    def __call__(
        self,
        predictions: Tensor,
        sky_mask: Tensor,
    ):
        # note that predictions should be weights if loss_type is weights_based
        # and opacity if loss_type is opacity_based
        loss = self.loss_fn(predictions, sky_mask)
        return self.return_loss(self.name, loss)


class DepthLoss(Loss):
    """
    Class for computing depth loss.

    Args:
        loss_type (Literal["l1", "l2", "smooth_l1"]): Type of loss to use.
        name (str): Name of the loss.
        normalize (bool): Whether to normalize the loss.
        depth_error_percentile (float): Percentile of depth values to use.
        coef (float): Coefficient to multiply the loss by.
        upper_bound (float): truncation value for the depth values.
        reduction (str): Reduction method for the loss.
        check_nan (bool): Whether to check for NaN values in the loss.

    Attributes:
        loss_type (Literal["l1", "l2", "smooth_l1"]): Type of loss being used.
        normalize (bool): Whether the loss is normalized.
        name (str): Name of the loss.
        upper_bound (float): Truncation value for the depth values.
        depth_error_percentile (float): Percentile of depth values being used.
    """

    def __init__(
        self,
        loss_type: Literal[
            "l1",
            "l2",
            "smooth_l1",
        ] = "l2",
        name: str = "depth_loss",
        normalize: bool = True,
        depth_error_percentile: float = None,
        coef: float = 1.0,
        upper_bound: float = 80,
        reduction="mean",
        check_nan=False,
    ):
        super(DepthLoss, self).__init__(coef, check_nan, reduction)
        self.loss_type = loss_type
        self.normalize = normalize
        self.name = f"{name}_{self.loss_type}"
        self.upper_bound = upper_bound
        self.depth_error_percentile = depth_error_percentile

    def _compute_depth_loss(
        self,
        pred_depth: Tensor,
        gt_depth: Tensor,
        max_depth: float = 80,
    ):
        pred_depth = pred_depth.squeeze()
        gt_depth = gt_depth.squeeze()
        valid_mask = (gt_depth > 0.01) & (gt_depth < max_depth)
        pred_depth = normalize_depth(pred_depth[valid_mask], max_depth=max_depth)
        gt_depth = normalize_depth(gt_depth[valid_mask], max_depth=max_depth)
        if self.loss_type == "smooth_l1":
            return F.smooth_l1_loss(pred_depth, gt_depth, reduction="none")
        elif self.loss_type == "l1":
            return F.l1_loss(pred_depth, gt_depth, reduction="none")
        elif self.loss_type == "l2":
            return F.mse_loss(pred_depth, gt_depth, reduction="none")
        else:
            raise NotImplementedError(f"Unknown loss type: {self.loss_type}")

    def __call__(
        self,
        pred_depth: Tensor,
        gt_depth: Tensor,
        name: str = None,
    ):
        depth_error = self._compute_depth_loss(pred_depth, gt_depth)
        if self.depth_error_percentile is not None:
            # to avoid outliers. not used for now
            depth_error = depth_error.flatten()
            depth_error = depth_error[
                depth_error.argsort()[
                    : int(len(depth_error) * self.depth_error_percentile)
                ]
            ]

        name = self.name if name is None else name
        return self.return_loss(name, depth_error)


class LineOfSightLoss(Loss):
    """
    Line of sight loss function.

    Args:
        loss_type (Literal["my",]): The type of loss to use.
        name (str): The name of the loss function.
        depth_error_percentile (float): The percentile of rays to optimize within each batch that have smallest depth error.
        coef (float): The coefficient to multiply the loss by.
        upper_bound (float): The upper bound of the loss.
        reduction (str): The reduction method to use.
        check_nan (bool): Whether to check for NaN values in the loss.

    Attributes:
        loss_type (Literal["my",]): The type of loss being used.
        name (str): The name of the loss function.
        upper_bound (float): The upper bound of the loss.
        depth_error_percentile (float): The percentile of rays to optimize within each batch that have smallest depth error.
    """

    def __init__(
        self,
        loss_type: Literal[
            "my",
        ] = "my",
        name: str = "line_of_sight",
        depth_error_percentile: float = None,
        coef: float = 1.0,
        upper_bound: float = 80,
        reduction="mean",
        check_nan=False,
    ):
        super(LineOfSightLoss, self).__init__(coef, check_nan, reduction)
        self.loss_type = loss_type
        self.name = f"{name}_{self.loss_type}"
        self.upper_bound = upper_bound
        self.depth_error_percentile = depth_error_percentile

    def __call__(
        self,
        pred_depth: Tensor,
        gt_depth: Tensor,
        weights: Tensor,
        t_vals: Tensor,
        epsilon: float,
        name: str = None,
        coef_decay: float = 1.0,
    ):
        if self.loss_type == "my":
            depth_error = compute_line_of_sight_loss(
                gt_depth, weights, t_vals.detach(), epsilon
            )
        else:
            raise NotImplementedError(f"Unknown loss type: {self.loss_type}")
        if self.depth_error_percentile is not None:
            depth_error = depth_error.flatten()
            depth_error = depth_error[
                depth_error.argsort()[
                    : int(len(depth_error) * self.depth_error_percentile)
                ]
            ]
        name = self.name if name is None else name
        depth_error = depth_error * coef_decay
        return self.return_loss(name, depth_error)


class DynamicRegularizationLoss(Loss):
    """
    A class representing a dynamic regularization loss function.

    Args:
        name (str): The name of the loss function.
        loss_type (Literal["sparsity", "entropy"]): The type of loss function to use.
        coef (float): The coefficient to multiply the loss by.
        entropy_skewness (float): The skewness factor for the entropy loss function.
        reduction (str): The reduction method to use for the loss.
        check_nan (bool): Whether to check for NaN values in the loss.

    Attributes:
        loss_type (Literal["sparsity", "entropy"]): The type of loss function to use.
        entropy_skewness (float): The skewness factor for the entropy loss function.
        name (str): The name of the loss function.

    Methods:
        __call__(self, dynamic_density: Tensor, static_density: Tensor = None, mask: Tensor = None, name: str = None):
            Computes the loss value for the given inputs.
    """

    def __init__(
        self,
        name: str = "dynamic",
        loss_type: Literal["sparsity", "entropy"] = "sparsity",
        coef: float = 1.0,
        entropy_skewness: float = 2.0,
        reduction="mean",
        check_nan=False,
    ):
        super(DynamicRegularizationLoss, self).__init__(coef, check_nan, reduction)
        self.loss_type = loss_type
        self.entropy_skewness = entropy_skewness
        self.name = f"{name}_{self.loss_type}_loss"

    def __call__(
        self,
        dynamic_density: Tensor,
        static_density: Tensor = None,
        mask: Tensor = None,
        name: str = None,
    ):
        """
        Computes the loss value for the given inputs.

        Args:
            dynamic_density (Tensor): The dynamic density tensor.
            static_density (Tensor): The static density tensor.
            mask (Tensor): The mask tensor.
            name (str): The name of the loss function.

        Returns:
            The computed loss value.

        """
        if self.loss_type == "sparsity":
            loss = dynamic_density
            if mask is not None:
                # further penalize the dynamic density of the rays that are within the mask
                loss = loss + 2 * dynamic_density * mask.unsqueeze(-1)
        elif self.loss_type == "entropy":
            # this loss didn't work well at first, and we didn't test it much since then
            dynamic_ratio = dynamic_density / (dynamic_density + static_density + 1e-7)
            dynamic_ratio_skewed = dynamic_ratio**self.entropy_skewness
            dynamic_ratio_skewed = dynamic_ratio_skewed.clamp(1e-6, 1 - 1e-6)
            dynamic_entropy_loss = (
                -(dynamic_ratio_skewed * dynamic_ratio_skewed.log())
                + -(1 - dynamic_ratio_skewed) * (1 - dynamic_ratio_skewed).log()
            )
            loss = dynamic_entropy_loss
        name = self.name if name is None else name
        return self.return_loss(name, loss)


def dirac_delta_approx(x, mu=0, sigma=1e-5):
    """
    Approximates the Dirac delta function with a Gaussian distribution.

    Args:
        x (torch.Tensor): The input tensor.
        mu (float, optional): The mean of the Gaussian distribution. Defaults to 0.
        sigma (float, optional): The standard deviation of the Gaussian distribution. Defaults to 1e-5.

    Returns:
        torch.Tensor: The output tensor.
    """
    return (1 / (math.sqrt(2 * torch.pi * sigma**2))) * torch.exp(
        -((x - mu) ** 2) / (2 * sigma**2)
    )


def compute_line_of_sight_loss(
    gt_depth: Tensor,
    weights: Tensor,
    t_vals: Tensor,
    epsilon: float = 2.0,
):
    """
    Computes the line-of-sight loss between the predicted and ground truth depth.

    Args:
        gt_depth (Tensor): Ground truth termination point.
        weights (Tensor): weights of each sampled interval.
        t_vals (Tensor): midpoint of each sampled interval.
        epsilon (float, optional): Margin for the line-of-sight loss. Defaults to 2.0.

    Returns:
        Tensor: Line-of-sight loss between the predicted and ground truth depth.
    """
    gt_depth, t_vals = gt_depth.squeeze(), t_vals.squeeze()
    depth_mask = gt_depth > 0
    gt_depth = gt_depth.unsqueeze(-1)

    empty_mask = t_vals < gt_depth - epsilon
    near_mask = (t_vals > (gt_depth - epsilon)) & (t_vals < gt_depth + epsilon)
    empty_loss = accumulate_along_rays(
        weights.square(), empty_mask.unsqueeze(-1)
    ).mean()
    near_loss = accumulate_along_rays(
        (weights - dirac_delta_approx(t_vals - gt_depth, sigma=epsilon / 3)).square(),
        near_mask.unsqueeze(-1),
    ).mean()
    # far_mask = t_vals > gt_depth + epsilon
    # far_loss = accumulate_along_rays(weights.square(), far_mask.unsqueeze(-1)).mean()
    sight_loss = empty_loss + near_loss  # + far_loss
    return sight_loss * depth_mask


class VisionDepthLoss(Loss):
    """
    Class for computing vision depth loss.

    Args:
        loss_type (Literal["l1", "l2", "smooth_l1"]): Type of loss to use.
        name (str): Name of the loss.
        coef (float): Coefficient to multiply the loss by.
        max_depth (float): truncation value for the depth values.
        reduction (str): Reduction method for the loss.
        check_nan (bool): Whether to check for NaN values in the loss.
    """

    def __init__(
        self,
        loss_type: Literal[
            "l1",
            "l2",
            "smooth_l1",
        ] = "l2",
        name: str = "vision_depth_loss",
        coef: float = 1.0,
        reduction="mean",
        max_depth: float = 1000,
        check_nan=False
    ):
        super(VisionDepthLoss, self).__init__(coef, check_nan, reduction)
        self.loss_type = loss_type
        self.name = f"{name}_{self.loss_type}"
        self.max_depth = max_depth

    def _compute_depth_loss(
        self,
        pred_depth: Tensor,
        gt_depth: Tensor
    ):
        pred_depth = pred_depth.squeeze()
        gt_depth = gt_depth.squeeze()
        valid_mask = (gt_depth > 0.0) & (gt_depth <= self.max_depth)
        pred_depth = normalize_depth(pred_depth[valid_mask], max_depth=self.max_depth)
        gt_depth = normalize_depth(gt_depth[valid_mask], max_depth=self.max_depth)

        if self.loss_type == "smooth_l1":
            return F.smooth_l1_loss(pred_depth, gt_depth, reduction="none")
        elif self.loss_type == "l1":
            return F.l1_loss(pred_depth, gt_depth, reduction="none")
        elif self.loss_type == "l2":
            return F.mse_loss(pred_depth, gt_depth, reduction="none")
        else:
            raise NotImplementedError(f"Unknown loss type: {self.loss_type}")

    def __call__(
        self,
        pred_depth: Tensor,
        gt_depth: Tensor,
        name: str = None,
    ):
        depth_error = self._compute_depth_loss(pred_depth, gt_depth)
        
        name = self.name if name is None else name

        return self.return_loss(name, depth_error)


class FeatureLoss(Loss):
    """
    A class representing the semantic feature loss function.

    Args:
        name (str): The name of the loss function.
        coef (float): The coefficient to multiply the loss by.
        reduction (str): The reduction method to use for the loss.
        check_nan (bool): Whether to check for NaN values in the loss.

    Attributes:
        name (str): The name of the loss function.
    """

    def __init__(
        self,
        name: str = "feature",
        coef: float = 1.0,
        reduction="none",
        check_nan=False
    ):
        super(FeatureLoss, self).__init__(coef, check_nan, reduction)
        self.name = f"{name}_loss"

    def __call__(
        self,
        semantic_feat: Tensor,
        selected_clip_vis_feat: Tensor,
        name: str = None,
    ):
        """
        Computes the cosine loss.

        Args:
            semantic_feat (Tensor): The rendered semantic features.
            selected_clip_vis_feat (Tensor): The selected scaled CLIP pixel features.
            name (str): The name of the loss function.

        Returns:
            dict: A dictionary containing the loss value.
        """
        loss = 1 - F.cosine_similarity(semantic_feat, selected_clip_vis_feat, dim=1).mean()
        
        name = self.name if name is None else name
        
        return self.return_loss(name, loss)


class FDALoss(Loss):
    """
    A class representing a feature distribution alignment loss function.

    Args:
        name (str): The name of the loss function.
        dino_shift (float): The DINO feature similarity threshold.
        coef (float): The coefficient to multiply the loss by.
        reduction (str): The reduction method to use for the loss.
        check_nan (bool): Whether to check for NaN values in the loss.
    """

    def __init__(
        self,
        name: str = "fda",
        dino_shift: float = 0.7,  # dino_shift = 0.7
        coef: float = 1.0,
        reduction="none",
        check_nan=False
    ):
        super(FDALoss, self).__init__(coef, check_nan, reduction)
        self.name = f"{name}_loss"
        self.dino_shift = dino_shift

    def tensor_correlation(self, a, b):
        return torch.einsum("nsc,nlc->nsl", F.normalize(a, dim=-1), F.normalize(b, dim=-1))

    def l1_correlation(self, a, b):
        '''
        not in log space!!!
        a, b: [N,HW,C]
        '''
        a = a.unsqueeze(2) # [N, HW, 1, C]
        b = b.unsqueeze(1) # [N, 1, HW, C]
        l1_corr = torch.abs(a-b).sum(dim=-1) # [N,HW,HW]

        return l1_corr

    def js_correlation(self, a, b):
        '''
        note that the input probabilities are already in log space
        a, b: [N,HW,C]
        '''
        m = torch.log((a.exp() + b.exp()) / 2.) # [N,HW,C]
        m = m.unsqueeze(1) # [N,1,HW,C]
        # KL(a||m), a is the true distribution
        a = a.unsqueeze(2) # [N,HW,1,C]
        kl_pointwise_am = a.exp() * (a-m) # [N,HW,HW,C]
        kl_am = kl_pointwise_am.sum(dim=-1) # [N,HW,HW]
        # KL(b||m), b is the true distribution
        b = b.unsqueeze(2)
        kl_pointwise_bm = b.exp() * (b-m)
        kl_bm = kl_pointwise_bm.sum(dim=-1)
        
        return (kl_am + kl_bm) / 2.

    def get_correlation_loss(self,
                feats: torch.Tensor,
                p_class: torch.Tensor,
    ):
        '''
        feats: [N,H,W,C]
        p_class: [N,H,W,N_class]
        '''
        feats = feats.reshape(feats.size(0), -1, feats.size(-1))
        p_class = p_class.reshape(p_class.size(0), -1, p_class.size(-1))  # h,w 2D => 1D，facilates subcequent pointwise similarity calculations

        with torch.no_grad():
            # get dino feature correlation
            f_corr = self.tensor_correlation(feats, feats) - self.dino_shift
            f_corr_pos = f_corr.clamp(min=0)
            f_corr_neg = f_corr.clamp(max=0)

        p_corr = self.js_correlation(p_class, p_class)

        return  (f_corr_pos * p_corr).sum() / torch.count_nonzero(f_corr_pos), \
                (f_corr_neg * p_corr).sum() / torch.count_nonzero(f_corr_neg)

    def __call__(self, dino_feature_map, log_p_class, dino_pos_weight, dino_neg_weight,
                 name: str = None):
        """
        Computes the feature distribution alignment loss.

        Args:
            dino_feature_map (Tensor): The DINO feature map.
            log_p_class (Tensor): The relevance map in logarithm space.
            dino_pos_weight (float): The weight for the positive correlation.
            dino_neg_weight (float): The weight for the negative correlation.
            name (str): The name of the loss function.
        Returns:
            dict: A dictionary containing the loss value.
        """
        pos_loss, neg_loss = self.get_correlation_loss(dino_feature_map, log_p_class)
        pos_loss = pos_loss * dino_pos_weight if dino_pos_weight != 0 else 0
        neg_loss = neg_loss * dino_neg_weight if dino_neg_weight != 0 else 0
        loss = pos_loss + neg_loss

        name = self.name if name is None else name

        return self.return_loss(name, loss)


class SRMRLoss(Loss):
    """
    The Semantic Relevancy Map Regularization Loss.

    Args:
        name (str): The name of the loss function.
        coef (float): The coefficient to multiply the loss by.
        reduction (str): The reduction method to use for the loss.
        check_nan (bool): Whether to check for NaN values in the loss.

    """

    def __init__(
        self,
        name: str = "srmr",
        coef: float = 1.0,
        reduction="mean",
        check_nan=False
    ):
        super(SRMRLoss, self).__init__(coef, check_nan, reduction)
        self.name = f"{name}_loss"
        self.loss_fn = F.cross_entropy

    def __call__(
        self,
        relevancy_map: Tensor,
        srmr_mask: Tensor,
        name: str = None,
    ):
        """
        Computes the srmr loss.

        Args:
            relevancy_map (Tensor): The relevancy map for the rendered semantic features.
            srmr_mask (Tensor): The semantically refined relevancy mask.
            name (str): The name of the loss function.

        Returns:
            dict: A dictionary containing the loss value.
        """
        loss = self.loss_fn(relevancy_map, srmr_mask, reduction="none", label_smoothing=0.1)
        
        name = self.name if name is None else name
        
        return self.return_loss(name, loss)


class CVSCLoss(Loss):
    """
    A class representing a Cross View Semantic Consistency Loss function.

    Args:
        name (str): The name of the loss function.
        coef (float): The coefficient to multiply the loss by.
        reduction (str): The reduction method to use for the loss.
        check_nan (bool): Whether to check for NaN values in the loss.
    """

    def __init__(
        self,
        name: str = "cvsc",
        coef: float = 1.0,
        reduction="none",
        check_nan=False
    ):
        super(CVSCLoss, self).__init__(coef, check_nan, reduction)
        self.name = f"{name}_loss"
    
    def get_srmat_loss(
        self,
        relevancy_map_train: Tensor,
        srmr_mask_train: Tensor
    ):
        """
        method to compute the Semantic Relevancy Map Approximation loss for training views.
        
        Args:
            relevancy_map_train (Tensor): The rendered relevancy map with clip text features for training views.
            srmr_mask_train (Tensor): The semantically refined relevancy map for training views.
        
        Returns:
            Tensor: The computed srmat loss value.
        """
        loss = F.cross_entropy(relevancy_map_train, srmr_mask_train, reduction="none",
                               label_smoothing=0.1)

        return loss.mean()

    def get_srman_loss(
        self,
        relevancy_map_pixel: Tensor,
        srmr_mask_pixel: Tensor
    ):
        """
        method to compute the Semantic Relevancy Map Approximation loss for novel views
        
        Args:
            relevancy_map_pixel (Tensor): The rendered relevancy map with clip text features for novel views.
            srmr_mask_pixel (Tensor): The semantically refined relevancy map for novel views.

        Returns:
            Tensor: The computed srman loss value.
        """
        loss = F.cross_entropy(relevancy_map_pixel, srmr_mask_pixel, reduction="none",
                               label_smoothing=0.1)
        
        return loss.mean()

    def __call__(
        self,
        relevancy_map_train: Tensor,
        relevancy_map_pixel: Tensor,
        srmr_mask_train: Tensor,
        srmr_mask_pixel: Tensor,
        name: str = None
    ):
        """
        Computes the cross view semantic consistency loss.

        Args:
            relevancy_map_train (Tensor): The rendered relevancy map with clip text features for training views.
            relevancy_map_pixel (Tensor): The rendered relevancy map with clip text features for novel views.
            srmr_mask_train (Tensor): The semantically refined relevancy map for training views.
            srmr_mask_pixel (Tensor): The semantically refined relevancy map for novel views.
            name (str): The name of the loss function.
        
        Returns:
            dict: A dictionary containing the loss value.
        """
        loss = self.get_srmat_loss(relevancy_map_train, srmr_mask_train) + \
               self.get_srman_loss(relevancy_map_pixel, srmr_mask_pixel)

        name = self.name if name is None else name

        return self.return_loss(name, loss)


class InstanceConsistencyLoss(Loss):
    """
    A class representing an instance consistency loss function.

    Args:
        name (str): The name of the loss function.
        coef (float): The coefficient to multiply the loss by.
        reduction (str): The reduction method to use for the loss.
        check_nan (bool): Whether to check for NaN values in the loss.
    """

    def __init__(
        self,
        name: str = "instance_consistency",
        coef: float = 1.0,
        reduction="none",
        check_nan=False,
    ):
        super(InstanceConsistencyLoss, self).__init__(coef, check_nan, reduction)
        self.name = f"{name}_loss"
    
    def __call__(
        self,
        instance_features: Tensor,
        gt: Tensor,
        confidences: Tensor,
        name: str = None
    ):
        """
        Computes the instance consistency loss.

        Args:
            instance_features (Tensor): The predicted instance features.
            gt (Tensor): The ground truth instance indices.
            confidences (Tensor): The confidence values.
            name (str): The name of the loss function.

        Returns:
            dict: A dictionary containing the loss value.
        """
        dim_instance_feature = instance_features.size(-1)
        fast_features, slow_features = instance_features.split(
            [dim_instance_feature // 2, dim_instance_feature // 2], dim=-1)
        slow_features = slow_features.detach() # no gradient for slow features

        # sample two random batches from the current batch
        fast_mask = torch.zeros_like(gt).bool()
        num_samples = gt.size(0) // 2
        high = gt.size(0)
        random_indices = torch.randint(0, high, (num_samples,)).to(gt.device)
        fast_mask[random_indices] = True
        slow_mask = ~fast_mask # non-overlapping masks for slow and fast models
        
        ## compute centroids
        slow_centroids = []
        fast_labels, slow_labels = torch.unique(gt[fast_mask]), torch.unique(gt[slow_mask])
        for l in slow_labels:
            mask_ = torch.logical_and(slow_mask, gt==l) #.unsqueeze(-1)
            slow_centroids.append(slow_features[mask_].mean(dim=0))
        slow_centroids = torch.stack(slow_centroids)

        # DEBUG edge case:
        if len(fast_labels) == 0 or len(slow_labels) == 0:
            print("Length of fast labels", len(fast_labels), "Length of slow labels", len(slow_labels))
            # This happens when gt of shape 1
            return torch.tensor(0.0, device=instance_features.device)

        ### Concentration loss Eq. (5)
        intersecting_labels = fast_labels[torch.where(torch.isin(fast_labels, slow_labels))] # [num_centroids]
        loss = 0
        for l in intersecting_labels:
            mask_ = torch.logical_and(fast_mask, gt==l)
            centroid_ = slow_centroids[slow_labels==l] # [1, d]
            # distance between fast features and slow centroid
            dist_sq = torch.pow(fast_features[mask_] - centroid_, 2).sum(dim=-1) # [num_points]
            loss += -1.0 * (torch.exp(-dist_sq / 1.0) * confidences[mask_]).mean()
        if intersecting_labels.shape[0] > 0: 
            loss /= intersecting_labels.shape[0]
        
        ### Contrastive loss Eq. (4)
        label_matrix = gt[fast_mask].unsqueeze(1) == gt[slow_mask].unsqueeze(0) # [num_points1, num_points2]
        similarity_matrix = torch.exp(-torch.cdist(fast_features[fast_mask], slow_features[slow_mask], p=2) / 1.0) # [num_points1, num_points2]
        logits = torch.exp(similarity_matrix)
        # compute loss
        prob = torch.mul(logits, label_matrix).sum(dim=-1) / logits.sum(dim=-1)
        prob_masked = torch.masked_select(prob, prob.ne(0))
        loss += -torch.log(prob_masked).mean()

        name = self.name if name is None else name

        return self.return_loss(name, loss)