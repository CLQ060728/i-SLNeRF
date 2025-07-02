# Author: Qian Liu
# Email: liu.qian.pro@gmail.com

import logging
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from radiance_fields.encodings import (
    HashEncoder,
    SinusoidalEncoder,
    build_xyz_encoder_from_cfg,
)
from radiance_fields.nerf_utils import contract, contract_inner, find_topk_nearby_timesteps, trunc_exp
from radiance_fields.mlp import MLP

logger = logging.getLogger()


class RadianceField(nn.Module):
    def __init__(
        self,
        xyz_encoder: HashEncoder,
        dynamic_xyz_encoder: Optional[HashEncoder] = None,
        flow_xyz_encoder: Optional[HashEncoder] = None,
        aabb: Union[Tensor, List[float]] = [-1, -1, -1, 1, 1, 1],
        num_dims: int = 3,
        density_activation: Callable = lambda x: trunc_exp(x - 1),
        unbounded: bool = True,
        contract_method: str = "aabb_bounded",
        inner_range: List[float] = [50.0, 50.0, 10.0],
        contract_ratio: float = 0.5,
        geometry_feature_dim: int = 15,
        base_mlp_layer_width: int = 64,
        head_mlp_layer_width: int = 64,
        enable_segmentation_head: bool = False,
        split_semantic_instance: bool = False,
        segmentation_feature_dim: int = 128,
        semantic_hidden_dim: int = 64,
        instance_hidden_dim: int = 64,
        selection_scale_dim: int = 3,
        semantic_embedding_dim: int = 128,
        instance_embedding_dim: int = 128,
        momentum: float = 0.9,
        enable_cam_embedding: bool = False,
        enable_img_embedding: bool = False,
        num_cams: int = 3,
        appearance_embedding_dim: int = 16,
        enable_sky_head: bool = False,
        enable_shadow_head: bool = False,
        num_train_timesteps: int = 0,
        interpolate_xyz_encoding: bool = False,
        enable_temporal_interpolation: bool = False,
    ) -> None:
        super().__init__()
        # scene properties
        if not isinstance(aabb, Tensor):
            aabb = torch.tensor(aabb, dtype=torch.float32)
        self.register_buffer("aabb", aabb)
        self.unbounded = unbounded
        self.contract_method = contract_method
        self.inner_range = torch.tensor(inner_range, dtype=torch.float32)
        self.contract_ratio = contract_ratio
        self.num_cams = num_cams
        self.num_dims = num_dims
        self.density_activation = density_activation
        self.momentum = momentum

        # appearance embedding
        self.enable_cam_embedding = enable_cam_embedding
        self.enable_img_embedding = enable_img_embedding
        self.appearance_embedding_dim = appearance_embedding_dim

        self.geometry_feature_dim = geometry_feature_dim

        if enable_segmentation_head:
            self.segmentation_feature_dim = segmentation_feature_dim
        else:
            segmentation_feature_dim = 0
            self.segmentation_feature_dim = segmentation_feature_dim

        # ======== Static Field ======== #
        self.xyz_encoder = xyz_encoder
        self.base_mlp = nn.Sequential(
            nn.Linear(self.xyz_encoder.n_output_dims, base_mlp_layer_width),
            nn.ReLU(),
            nn.Linear(
                base_mlp_layer_width, geometry_feature_dim + segmentation_feature_dim
            ),
        )

        # ======== Dynamic Field ======== #
        self.interpolate_xyz_encoding = interpolate_xyz_encoding
        self.dynamic_xyz_encoder = dynamic_xyz_encoder
        self.enable_temporal_interpolation = enable_temporal_interpolation
        if self.dynamic_xyz_encoder is not None:
            # for temporal interpolation
            self.register_buffer("training_timesteps", torch.zeros(num_train_timesteps))
            self.dynamic_base_mlp = nn.Sequential(
                nn.Linear(self.dynamic_xyz_encoder.n_output_dims, base_mlp_layer_width),
                nn.ReLU(),
                nn.Linear(
                    base_mlp_layer_width,
                    geometry_feature_dim + segmentation_feature_dim
                ),
            )

        # ======== Flow Field ======== #
        self.flow_xyz_encoder = flow_xyz_encoder
        if self.flow_xyz_encoder is not None:
            self.flow_mlp = nn.Sequential(
                nn.Linear(
                    self.flow_xyz_encoder.n_output_dims,
                    base_mlp_layer_width,
                ),
                nn.ReLU(),
                nn.Linear(base_mlp_layer_width, base_mlp_layer_width),
                nn.ReLU(),
                nn.Linear(base_mlp_layer_width, 6),  # 3 for forward, 3 for backward
                # no activation function for flow
            )

        # appearance embedding
        if self.enable_cam_embedding:
            # per-camera embedding
            self.appearance_embedding = nn.Embedding(num_cams, appearance_embedding_dim)
        elif self.enable_img_embedding:
            # per-image embedding
            self.appearance_embedding = nn.Embedding(
                num_train_timesteps * num_cams, appearance_embedding_dim
            )
        else:
            self.appearance_embedding = None

        # direction encoding
        self.direction_encoding = SinusoidalEncoder(
            n_input_dims=3, min_deg=0, max_deg=4
        )

        # ======== Color Head ======== #
        self.rgb_head = MLP(
            in_dims=geometry_feature_dim
            + self.direction_encoding.n_output_dims
            + (
                appearance_embedding_dim
                if self.enable_cam_embedding or self.enable_img_embedding
                else 0  # 2 or 0?
            ),
            out_dims=3,
            num_layers=3,
            hidden_dims=head_mlp_layer_width,
            skip_connections=[1],
        )

        # ======== Shadow Head ======== #
        self.enable_shadow_head = enable_shadow_head
        if self.enable_shadow_head:
            self.shadow_head = nn.Sequential(
                nn.Linear(geometry_feature_dim, base_mlp_layer_width),
                nn.ReLU(),
                nn.Linear(base_mlp_layer_width, 1),
                nn.Sigmoid(),
            )

        # ======== Sky Head ======== #
        self.enable_sky_head = enable_sky_head
        if self.enable_sky_head:
            self.sky_head = MLP(
                in_dims=self.direction_encoding.n_output_dims
                + (
                    appearance_embedding_dim
                    if self.enable_cam_embedding or self.enable_img_embedding
                    else 0
                ),
                out_dims=3,
                num_layers=3,
                hidden_dims=head_mlp_layer_width,
                skip_connections=[1],
            )
            if enable_segmentation_head:
                if split_semantic_instance:
                    # split semantic and instance branches
                    self.semantic_sky_head = nn.Sequential(
                        nn.Linear(
                            self.direction_encoding.n_output_dims,
                            semantic_hidden_dim,
                        ),
                        nn.ReLU(),
                        nn.Linear(semantic_hidden_dim, semantic_hidden_dim * 2),
                        nn.ReLU(),
                        nn.Linear(semantic_hidden_dim * 2, semantic_embedding_dim)
                    )
                    self.instance_sky_head = nn.Sequential(
                        nn.Linear(
                            self.direction_encoding.n_output_dims,
                            instance_hidden_dim,
                        ),
                        nn.ReLU(),
                        nn.Linear(instance_hidden_dim, instance_hidden_dim * 2),
                        nn.ReLU(),
                        nn.Linear(instance_hidden_dim * 2, instance_embedding_dim)
                    )
                else:
                    # shared branch for semantic and instance
                    self.instance_sky_head = nn.Sequential(
                        nn.Linear(
                            self.direction_encoding.n_output_dims,
                            instance_hidden_dim,
                        ),
                        nn.ReLU(),
                        nn.Linear(instance_hidden_dim, instance_hidden_dim * 2),
                        nn.ReLU(),
                        nn.Linear(instance_hidden_dim * 2, instance_embedding_dim)
                    )
                    self.semantic_sky_head = nn.Sequential(
                        nn.Linear(
                            instance_embedding_dim,
                            semantic_hidden_dim,
                        ),
                        nn.ReLU(),
                        nn.Linear(semantic_hidden_dim, semantic_embedding_dim)
                    )

        # ======== Segmentation Head ======== #
        self.enable_segmentation_head = enable_segmentation_head
        self.split_semantic_instance = split_semantic_instance
        if self.enable_segmentation_head:
            if self.split_semantic_instance:
                # selection head for semantic clip feature selection
                self.selection_head = nn.Sequential(
                    nn.linear(segmentation_feature_dim // 2, semantic_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(semantic_hidden_dim, semantic_hidden_dim * 2),
                    nn.ReLU(),
                    nn.Linear(semantic_hidden_dim * 2, selection_scale_dim)
                )
                # split semantic and instance branches
                self.semantic_head = nn.Sequential(
                nn.Linear(segmentation_feature_dim // 2, semantic_hidden_dim),
                nn.ReLU(),
                nn.Linear(semantic_hidden_dim, semantic_hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(semantic_hidden_dim * 2, semantic_embedding_dim)
                )
                self.fast_instance_head = nn.Sequential(
                    nn.Linear(segmentation_feature_dim // 2, instance_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(instance_hidden_dim, instance_hidden_dim * 2),
                    nn.ReLU(),
                    nn.Linear(instance_hidden_dim * 2, instance_embedding_dim // 2)
                )
                self.slow_instance_head = nn.Sequential(
                    nn.Linear(segmentation_feature_dim // 2, instance_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(instance_hidden_dim, instance_hidden_dim * 2),
                    nn.ReLU(),
                    nn.Linear(instance_hidden_dim * 2, instance_embedding_dim  // 2)
                )
            else:
                # shared branch for semantic and instance
                self.fast_instance_head = nn.Sequential(
                    nn.Linear(segmentation_feature_dim, instance_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(instance_hidden_dim, instance_hidden_dim * 2),
                    nn.ReLU(),
                    nn.Linear(instance_hidden_dim * 2, instance_embedding_dim // 2)
                )
                self.slow_instance_head = nn.Sequential(
                    nn.Linear(segmentation_feature_dim, instance_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(instance_hidden_dim, instance_hidden_dim * 2),
                    nn.ReLU(),
                    nn.Linear(instance_hidden_dim * 2, instance_embedding_dim // 2)
                )
                self.semantic_head = nn.Sequential(
                    nn.Linear(instance_embedding_dim, semantic_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(semantic_hidden_dim, semantic_embedding_dim)
                )
                self.selection_head = nn.Sequential(
                    nn.Linear(segmentation_feature_dim, semantic_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(semantic_hidden_dim, semantic_hidden_dim * 2),
                    nn.ReLU(),
                    nn.Linear(semantic_hidden_dim * 2, selection_scale_dim)
                )


    def register_normalized_training_timesteps(
        self, normalized_timesteps: Tensor, time_diff: float = None
    ) -> None:
        """
        register normalized timesteps for temporal interpolation

        Args:
            normalized_timesteps (Tensor): normalized timesteps in [0, 1]
            time_diff (float, optional): time difference between two consecutive timesteps. Defaults to None.
        """
        if self.dynamic_xyz_encoder is not None:
            # register timesteps for temporal interpolation
            self.training_timesteps.copy_(normalized_timesteps)
            self.training_timesteps = self.training_timesteps.to(self.device)
            if time_diff is not None:
                # use the provided time difference if available
                self.time_diff = time_diff
            else:
                if len(self.training_timesteps) > 1:
                    # otherwise, compute the time difference from the provided timesteps
                    # it's important to make sure the provided timesteps are consecutive
                    self.time_diff = (
                        self.training_timesteps[1] - self.training_timesteps[0]
                    )
                else:
                    self.time_diff = 0

    def set_aabb(self, aabb: Union[Tensor, List[float]]) -> None:
        """
        register aabb for scene space
        """
        if not isinstance(aabb, Tensor):
            aabb = torch.tensor(aabb, dtype=torch.float32)
        logger.info(f"Set aabb from {self.aabb} to {aabb}")
        self.aabb.copy_(aabb)
        self.aabb = self.aabb.to(self.device)

    @property
    def device(self) -> torch.device:
        return self.aabb.device

    def contract_points(
        self,
        positions: Tensor,
    ) -> Tensor:
        """
        contract [-inf, inf] points to the range [0, 1] for hash encoding

        Returns:
            normed_positions: [..., 3] in [0, 1]
        """
        if self.unbounded:
            if self.contract_method == "aabb_bounded":
                # use infinte norm to contract the positions for cuboid aabb
                normed_positions = contract(positions, self.aabb, ord=float("inf"))
            elif self.contract_method == "inner_outer":
                # use inner range to contract the positions for cuboid aabb
                normed_positions = contract_inner(positions, self.aabb, self.inner_range,
                                                  self.contract_ratio)
            else:
                raise NotImplementedError(
                    f"Contract method {self.contract_method} is not implemented."
                )
        else:
            aabb_min, aabb_max = torch.split(self.aabb, 3, dim=-1)
            normed_positions = (positions - aabb_min) / (aabb_max - aabb_min)
        selector = (
            ((normed_positions > 0.0) & (normed_positions < 1.0))
            .all(dim=-1)
            .to(positions)
        )
        normed_positions = normed_positions * selector.unsqueeze(-1)
        return normed_positions

    def ema_update_slownet(self) -> None:
        # EMA update for the slow network
        with torch.no_grad():
            for param_fast, param_slow in zip(self.fast_instance_head.parameters(),
                                              self.slow_instance_head.parameters()):
                param_slow.data.mul_(self.momentum).add_((1 - self.momentum) * param_fast.detach())


    def forward_static_hash(
        self,
        positions: Tensor,
    ) -> Tensor:
        """
        forward pass for static hash encoding

        Returns:
            encoded_features: [..., geometry_feature_dim]
            normed_positions: [..., 3] in [0, 1]
        """
        normed_positions = self.contract_points(positions)
        xyz_encoding = self.xyz_encoder(normed_positions.view(-1, self.num_dims))
        encoded_features = self.base_mlp(xyz_encoding).view(
            list(normed_positions.shape[:-1]) + [-1]
        )
        return encoded_features, normed_positions

    def forward_dynamic_hash(
        self,
        normed_positions: Tensor,
        normed_timestamps: Tensor,
        return_hash_encodings: bool = False,
    ) -> Union[Tuple[Tensor, Tensor], Tensor]:
        """
        forward pass for dynamic hash encoding

        Returns:
            encoded_dynamic_feats: [..., geometry_feature_dim]
            dynamic_xyz_encoding: [..., n_output_dims] (optional)
        """
        if normed_timestamps.shape[-1] != 1:
            normed_timestamps = normed_timestamps.unsqueeze(-1)
        # To be fixed.
        # if self.training or not self.enable_temporal_interpolation:
        if True:
            temporal_positions = torch.cat(
                [normed_positions, normed_timestamps], dim=-1
            )
            dynamic_xyz_encoding = self.dynamic_xyz_encoder(
                temporal_positions.view(-1, self.num_dims + 1)
            ).view(list(temporal_positions.shape[:-1]) + [-1])
            encoded_dynamic_feats = self.dynamic_base_mlp(dynamic_xyz_encoding)
        else:
            encoded_dynamic_feats = temporal_interpolation(
                normed_timestamps,
                self.training_timesteps,
                normed_positions,
                self.dynamic_xyz_encoder,
                self.dynamic_base_mlp,
                interpolate_xyz_encoding=self.interpolate_xyz_encoding,
            )
        if return_hash_encodings:
            return encoded_dynamic_feats, dynamic_xyz_encoding
        else:
            return encoded_dynamic_feats

    def forward_flow_hash(
        self,
        normed_positions: Tensor,
        normed_timestamps: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        forward pass for flow hash encoding

        Returns:
            flow: [..., 6] (forward_flow, backward_flow)
        """
        if normed_timestamps.shape[-1] != 1:
            normed_timestamps = normed_timestamps.unsqueeze(-1)
        if self.training or not self.enable_temporal_interpolation:
            temporal_positions = torch.cat(
                [normed_positions, normed_timestamps], dim=-1
            )
            flow_xyz_encoding = self.flow_xyz_encoder(
                temporal_positions.view(-1, self.num_dims + 1)
            ).view(list(temporal_positions.shape[:-1]) + [-1])
            flow = self.flow_mlp(flow_xyz_encoding)
        else:
            flow = temporal_interpolation(
                normed_timestamps,
                self.training_timesteps,
                normed_positions,
                self.flow_xyz_encoder,
                self.flow_mlp,
                interpolate_xyz_encoding=True,
            )
        return flow

    def forward(
        self,
        positions: Tensor,
        directions: Tensor = None,
        data_dict: Dict[str, Tensor] = {},
        return_density_only: bool = False,
        combine_static_dynamic: bool = False
    ) -> Dict[str, Tensor]:
        """
        Args:
            positions: [..., 3]
            directions: [..., 3]
            data_dict: a dictionary containing additional data
            return_density_only: if True, only return density without querying other heads
            combine_static_dynamic: if True, combine static and dynamic predictions based on static and dynamic density
            in addition to returning separate results for static and dynamic fields
        Returns:
            results_dict: a dictionary containing everything
        """
        results_dict = {}
        # forward static branch
        encoded_features, normed_positions = self.forward_static_hash(positions)
        if self.enable_segmentation_head:
            if self.split_semantic_instance:
                # split semantic and instance branches
                geo_feats, semantic_feats, instance_feats = torch.split(
                    encoded_features,
                    [self.geometry_feature_dim, self.segmentation_feature_dim // 2,
                     self.segmentation_feature_dim // 2],
                    dim=-1,
                )
            else:
                # shared branch for semantic and instance
                geo_feats, segmentation_feats = torch.split(
                    encoded_features,
                    [self.geometry_feature_dim, self.segmentation_feature_dim],
                    dim=-1,
                )
        else:
            geo_feats = torch.split(
            encoded_features,
            [self.geometry_feature_dim],
            dim=-1,
        )[0]
        static_density = self.density_activation(geo_feats[..., 0])

        has_timestamps = (
            "normed_timestamps" in data_dict or "lidar_normed_timestamps" in data_dict
        )
        if self.dynamic_xyz_encoder is not None and has_timestamps:
            # forward dynamic branch
            if "normed_timestamps" in data_dict:
                normed_timestamps = data_dict["normed_timestamps"]
            elif "lidar_normed_timestamps" in data_dict:
                # we use `lidar_` prefix as an identifier to skip querying other heads
                normed_timestamps = data_dict["lidar_normed_timestamps"]
            dynamic_feats, dynamic_hash_encodings = self.forward_dynamic_hash(
                normed_positions, normed_timestamps, return_hash_encodings=True
            )
            if self.flow_xyz_encoder is not None:
                flow = self.forward_flow_hash(normed_positions, normed_timestamps)
                forward_flow, backward_flow = flow[..., :3], flow[..., 3:]
                results_dict["forward_flow"] = forward_flow
                results_dict["backward_flow"] = backward_flow
                temporal_aggregation_results = self.temporal_aggregation(
                    positions,
                    normed_timestamps,
                    forward_flow,
                    backward_flow,
                    dynamic_feats,
                )
                # overwrite dynamic feats using temporal aggregation results
                dynamic_feats = temporal_aggregation_results["dynamic_feats"]
                
                temporal_aggregation_results[
                    "current_dynamic_hash_encodings"
                ] = dynamic_hash_encodings
                results_dict.update(temporal_aggregation_results)
            
            if self.enable_segmentation_head:
                if self.split_semantic_instance:
                    # split semantic and instance branches
                    dynamic_geo_feats, dynamic_semantic_feats, dynamic_instance_feats = torch.split(
                        dynamic_feats,
                        [self.geometry_feature_dim, self.segmentation_feature_dim // 2,
                         self.segmentation_feature_dim // 2],
                        dim=-1,
                    )
                else:
                    # shared branch for semantic and instance
                    dynamic_geo_feats, dynamic_segmentation_feats = torch.split(
                        dynamic_feats,
                        [self.geometry_feature_dim, self.segmentation_feature_dim],
                        dim=-1,
                    )
            else:
                dynamic_geo_feats = torch.split(
                    dynamic_feats,
                    [self.geometry_feature_dim],
                    dim=-1,
                )[0]
            dynamic_density = self.density_activation(dynamic_geo_feats[..., 0])
            # blend static and dynamic density to get the final density
            density = static_density + dynamic_density
            results_dict.update(
                {
                    "density": density,
                    "static_density": static_density,
                    "dynamic_density": dynamic_density,
                }
            )
            if return_density_only:
                # skip querying other heads
                return results_dict

            if directions is not None:
                rgb_results = self.query_rgb(
                    directions, geo_feats, dynamic_geo_feats, data_dict=data_dict
                )
                results_dict["dynamic_rgb"] = rgb_results["dynamic_rgb"]
                results_dict["static_rgb"] = rgb_results["rgb"]
                if combine_static_dynamic:
                    static_ratio = static_density / (density + 1e-6)
                    dynamic_ratio = dynamic_density / (density + 1e-6)
                    results_dict["rgb"] = (
                        static_ratio[..., None] * results_dict["static_rgb"]
                        + dynamic_ratio[..., None] * results_dict["dynamic_rgb"]
                    )
            if self.enable_shadow_head:
                shadow_ratio = self.shadow_head(dynamic_geo_feats)
                results_dict["shadow_ratio"] = shadow_ratio
                if combine_static_dynamic and "rgb" in results_dict:
                    results_dict["rgb"] = (
                        static_ratio[..., None]
                        * results_dict["rgb"]
                        * (1 - shadow_ratio)
                        + dynamic_ratio[..., None] * results_dict["dynamic_rgb"]
                    )
        else:
            # if no dynamic branch, use static density
            results_dict["density"] = static_density
            if return_density_only:
                # skip querying other heads
                return results_dict
            if directions is not None:
                rgb_results = self.query_rgb(directions, geo_feats, data_dict=data_dict)
                results_dict["rgb"] = rgb_results["rgb"]

        if self.enable_segmentation_head:
            if self.split_semantic_instance:
                # split semantic and instance branches
                selection_feats = self.selection_head(semantic_feats)
                logger.info(f"Selection features shape: {selection_feats.size()}")
                selection_mask = F.softmax(selection_feats, dim=-2)
                semantic_embedding = self.semantic_head(semantic_feats)
                semantic_embedding = F.normalize(semantic_embedding, dim=-1)
                fast_instance_embedding = self.fast_instance_head(instance_feats)
                slow_instance_embedding = self.slow_instance_head(instance_feats)
                fast_instance_embedding = F.normalize(fast_instance_embedding, dim=-1)
                slow_instance_embedding = F.normalize(slow_instance_embedding, dim=-1)
                instance_embedding = torch.cat([fast_instance_embedding, slow_instance_embedding], dim=-1)
            else:
                # shared branch for semantic and instance
                selection_feats = self.selection_head(segmentation_feats)
                logger.info(f"Selection features shape: {selection_feats.size()}")
                selection_mask = F.softmax(selection_feats, dim=-2)
                fast_instance_embedding = self.fast_instance_head(segmentation_feats)
                slow_instance_embedding = self.slow_instance_head(segmentation_feats)
                instance_embedding = torch.cat([fast_instance_embedding, slow_instance_embedding], dim=-1)
                semantic_embedding = self.semantic_head(instance_embedding)
                semantic_embedding = F.normalize(semantic_embedding, dim=-1)
                fast_instance_embedding = F.normalize(fast_instance_embedding, dim=-1)
                slow_instance_embedding = F.normalize(slow_instance_embedding, dim=-1)
                instance_embedding = torch.cat([fast_instance_embedding, slow_instance_embedding], dim=-1)
            
            if self.dynamic_xyz_encoder is not None and has_timestamps:
                if self.split_semantic_instance:
                    dynamic_selection_feats = self.selection_head(dynamic_semantic_feats)
                    logger.info(f"Dynamic selection features shape: {dynamic_selection_feats.size()}")
                    dynamic_selection_mask = F.softmax(dynamic_selection_feats, dim=-2)
                    dynamic_semantic_embedding = self.semantic_head(dynamic_semantic_feats)
                    dynamic_semantic_embedding = F.normalize(dynamic_semantic_embedding, dim=-1)
                    dynamic_fast_instance_embedding = self.fast_instance_head(dynamic_instance_feats)
                    dynamic_slow_instance_embedding = self.slow_instance_head(dynamic_instance_feats)
                    dynamic_fast_instance_embedding = F.normalize(dynamic_fast_instance_embedding, dim=-1)
                    dynamic_slow_instance_embedding = F.normalize(dynamic_slow_instance_embedding, dim=-1)
                    dynamic_instance_embedding = torch.cat(
                        [dynamic_fast_instance_embedding, dynamic_slow_instance_embedding], dim=-1
                    )
                else:
                    dynamic_selection_feats = self.selection_head(dynamic_segmentation_feats)
                    logger.info(f"Dynamic selection features shape: {dynamic_selection_feats.size()}")
                    dynamic_selection_mask = F.softmax(dynamic_selection_feats, dim=-2)
                    dynamic_fast_instance_embedding = self.fast_instance_head(dynamic_segmentation_feats)
                    dynamic_slow_instance_embedding = self.slow_instance_head(dynamic_segmentation_feats)
                    dynamic_instance_embedding = torch.cat(
                        [dynamic_fast_instance_embedding, dynamic_slow_instance_embedding], dim=-1
                    )
                    dynamic_semantic_embedding = self.semantic_head(dynamic_instance_embedding)
                    dynamic_semantic_embedding = F.normalize(dynamic_semantic_embedding, dim=-1)
                    dynamic_fast_instance_embedding = F.normalize(dynamic_fast_instance_embedding, dim=-1)
                    dynamic_slow_instance_embedding = F.normalize(dynamic_slow_instance_embedding, dim=-1)
                    dynamic_instance_embedding = torch.cat(
                        [dynamic_fast_instance_embedding, dynamic_slow_instance_embedding], dim=-1
                    )
                
                results_dict["static_selection_mask"] = selection_mask
                results_dict["static_semantic_embedding"] = semantic_embedding
                results_dict["static_instance_embedding"] = instance_embedding
                results_dict["dynamic_selection_mask"] = dynamic_selection_mask
                results_dict["dynamic_semantic_embedding"] = dynamic_semantic_embedding
                results_dict["dynamic_instance_embedding"] = dynamic_instance_embedding
                if combine_static_dynamic:
                    static_ratio = static_density / (density + 1e-6)
                    dynamic_ratio = dynamic_density / (density + 1e-6)
                    results_dict["selection_mask"] = (
                        static_ratio[..., None] * selection_mask
                        + dynamic_ratio[..., None] * dynamic_selection_mask
                    )
                    results_dict["semantic_embedding"] = (
                        static_ratio[..., None] * semantic_embedding
                        + dynamic_ratio[..., None] * dynamic_semantic_embedding
                    )
                    results_dict["instance_embedding"] = (
                        static_ratio[..., None] * instance_embedding
                        + dynamic_ratio[..., None] * dynamic_instance_embedding
                    )
            else:
                results_dict["selection_mask"] = selection_mask
                results_dict["semantic_embedding"] = semantic_embedding
                results_dict["instance_embedding"] = instance_embedding
                
        # query sky if not in lidar mode
        if (
            self.enable_sky_head
            and "lidar_origin" not in data_dict
            and directions is not None
        ):
            directions = directions[:, 0]
            reduced_data_dict = {k: v[:, 0] for k, v in data_dict.items()}
            sky_results = self.query_sky(directions, data_dict=reduced_data_dict)
            results_dict.update(sky_results)
        
        return results_dict


    def temporal_aggregation(
        self,
        positions: Tensor,  # current world coordinates
        normed_timestamps: Tensor,  # current normalized timestamps
        forward_flow: Tensor,
        backward_flow: Tensor,
        dynamic_feats: Tensor,
    ) -> Tensor:
        """
        temporal aggregation for dynamic features
        Eq. (8) in the emernerf paper
        """
        if normed_timestamps.shape[-1] != 1:
            normed_timestamps = normed_timestamps.unsqueeze(-1)
        if self.training:
            noise = torch.rand_like(forward_flow)[..., 0:1]
        else:
            noise = torch.ones_like(forward_flow)[..., 0:1]
        # forward and backward warped positions
        forward_warped_positions = self.contract_points(
            positions + forward_flow * noise
        )
        backward_warped_positions = self.contract_points(
            positions + backward_flow * noise
        )
        # forward and backward warped timestamps
        forward_warped_time = torch.clamp(
            normed_timestamps + self.time_diff * noise, 0, 1.0
        )
        backward_warped_time = torch.clamp(
            normed_timestamps - self.time_diff * noise, 0, 1.0
        )
        (
            forward_dynamic_feats,
            forward_dynamic_hash_encodings,
        ) = self.forward_dynamic_hash(
            forward_warped_positions,
            forward_warped_time,
            return_hash_encodings=True,
        )
        (
            backward_dynamic_feats,
            backward_dynamic_hash_encodings,
        ) = self.forward_dynamic_hash(
            backward_warped_positions,
            backward_warped_time,
            return_hash_encodings=True,
        )
        forward_pred_flow = self.forward_flow_hash(
            forward_warped_positions,
            forward_warped_time,
        )
        backward_pred_flow = self.forward_flow_hash(
            backward_warped_positions,
            backward_warped_time,
        )
        # simple weighted sum
        aggregated_dynamic_feats = (
            dynamic_feats + 0.5 * forward_dynamic_feats + 0.5 * backward_dynamic_feats
        ) / 2.0
        return {
            "dynamic_feats": aggregated_dynamic_feats,
            "forward_pred_backward_flow": forward_pred_flow[..., 3:],
            "backward_pred_forward_flow": backward_pred_flow[..., :3],
            
            "forward_dynamic_hash_encodings": forward_dynamic_hash_encodings,
            "backward_dynamic_hash_encodings": backward_dynamic_hash_encodings,
        }

    def query_rgb(
        self,
        directions: Tensor,
        geo_feats: Tensor,
        dynamic_geo_feats: Tensor = None,
        data_dict: Dict[str, Tensor] = None,
    ) -> Tensor:
        directions = (directions + 1.0) / 2.0  
        h = self.direction_encoding(directions.reshape(-1, directions.shape[-1])).view(
            *directions.shape[:-1], -1
        )
        if self.enable_cam_embedding or self.enable_img_embedding:
            if "cam_idx" in data_dict and self.enable_cam_embedding:
                appearance_embedding = self.appearance_embedding(data_dict["cam_idx"])
            elif "img_idx" in data_dict and self.enable_img_embedding:
                appearance_embedding = self.appearance_embedding(data_dict["img_idx"])
            else:
                # use mean appearance embedding
                # print("using mean appearance embedding")
                appearance_embedding = torch.ones(
                    (*directions.shape[:-1], self.appearance_embedding_dim),
                    device=directions.device,
                ) * self.appearance_embedding.weight.mean(dim=0)
            h = torch.cat([h, appearance_embedding], dim=-1)

        rgb = self.rgb_head(torch.cat([h, geo_feats], dim=-1))
        rgb = F.sigmoid(rgb)
        results = {"rgb": rgb}

        if self.dynamic_xyz_encoder is not None:
            assert (
                dynamic_geo_feats is not None
            ), "Dynamic geometry features are not provided."
            dynamic_rgb = self.rgb_head(torch.cat([h, dynamic_geo_feats], dim=-1))
            dynamic_rgb = F.sigmoid(dynamic_rgb)
            results["dynamic_rgb"] = dynamic_rgb
        return results

    def query_sky(
        self, directions: Tensor, data_dict: Dict[str, Tensor] = None
    ) -> Dict[str, Tensor]:
        if len(directions.shape) == 2:
            dd = self.direction_encoding(directions).to(directions)
        else:
            dd = self.direction_encoding(directions[:, 0]).to(directions)
        if self.enable_cam_embedding or self.enable_img_embedding:
            # optionally add appearance embedding
            if "cam_idx" in data_dict and self.enable_cam_embedding:
                appearance_embedding = self.appearance_embedding(data_dict["cam_idx"])
            elif "img_idx" in data_dict and self.enable_img_embedding:
                appearance_embedding = self.appearance_embedding(data_dict["img_idx"])
            else:
                # use mean appearance embedding
                appearance_embedding = torch.ones(
                    (*directions.shape[:-1], self.appearance_embedding_dim),
                    device=directions.device,
                ) * self.appearance_embedding.weight.mean(dim=0)
            dd = torch.cat([dd, appearance_embedding], dim=-1)
        rgb_sky = self.sky_head(dd).to(directions)
        rgb_sky = F.sigmoid(rgb_sky)
        results = {"rgb_sky": rgb_sky}

        if self.enable_segmentation_head:
            semantic_sky_embedding = self.semantic_sky_head(dd).to(directions)
            instance_sky_embedding = self.instance_sky_head(dd).to(directions)
            results["semantic_sky_embedding"] = F.sigmoid(semantic_sky_embedding)
            # results["semantic_sky_embedding"] = semantic_sky_embedding
            results["instance_sky_embedding"] = F.sigmoid(instance_sky_embedding)
        
        return results

    def query_flow(
        self, positions: Tensor, normed_timestamps: Tensor, query_density: bool = True
    ) -> Dict[str, Tensor]:
        """
        query flow field
        """
        normed_positions = self.contract_points(positions)
        flow = self.forward_flow_hash(normed_positions, normed_timestamps)
        results = {
            "forward_flow": flow[..., :3],
            "backward_flow": flow[..., 3:],
        }
        if query_density:
            # it's important to filter valid flows based on a dynamic density threshold.
            # flows are valid only if they are on dynamic points.
            dynamic_feats = self.forward_dynamic_hash(
                normed_positions, normed_timestamps
            )
            (dynamic_geo_feats, _,) = torch.split(
                dynamic_feats,
                [self.geometry_feature_dim, self.semantic_feature_dim],
                dim=-1,
            )
            dynamic_density = self.density_activation(dynamic_geo_feats[..., 0])
            results["dynamic_density"] = dynamic_density
        return results


class DensityField(nn.Module):
    def __init__(
        self,
        xyz_encoder: HashEncoder,
        aabb: Union[Tensor, List[float]] = [[-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]],
        num_dims: int = 3,
        density_activation: Callable = lambda x: trunc_exp(x - 1),
        unbounded: bool = False,
        contract_method = "aabb_bounded",
        inner_range: List[float] = [50.0, 50.0, 10.0],
        contract_ratio: float = 0.5,
        base_mlp_layer_width: int = 64,
    ) -> None:
        super().__init__()
        if not isinstance(aabb, Tensor):
            aabb = torch.tensor(aabb, dtype=torch.float32)
        self.register_buffer("aabb", aabb)
        self.num_dims = num_dims
        self.density_activation = density_activation
        self.unbounded = unbounded
        self.contract_method = contract_method
        self.inner_range = torch.tensor(inner_range, dtype=torch.float32)
        self.contract_ratio = contract_ratio
        self.xyz_encoder = xyz_encoder

        # density head
        self.base_mlp = nn.Sequential(
            nn.Linear(self.xyz_encoder.n_output_dims, base_mlp_layer_width),
            nn.ReLU(),
            nn.Linear(base_mlp_layer_width, 1),
        )

    @property
    def device(self) -> torch.device:
        return self.aabb.device

    def set_aabb(self, aabb: Union[Tensor, List[float]]) -> None:
        if not isinstance(aabb, Tensor):
            aabb = torch.tensor(aabb, dtype=torch.float32)
        logger.info(f"Set propnet aabb from {self.aabb} to {aabb}")
        self.aabb.copy_(aabb)
        self.aabb = self.aabb.to(self.device)

    def forward(
        self, positions: Tensor, data_dict: Dict[str, Tensor] = None
    ) -> Dict[str, Tensor]:
        if self.unbounded:
            if self.contract_method == "aabb_bounded":
                # use infinte norm to contract the positions for cuboid aabb
                positions = contract(positions, self.aabb, ord=float("inf"))
            elif self.contract_method == "inner_outer":
                # use inner range to contract the positions for cuboid aabb
                positions = contract_inner(positions, self.aabb, self.inner_range, self.contract_ratio)
            else:
                raise NotImplementedError(
                    f"Contract method {self.contract_method} is not implemented."
                )
        else:
            aabb_min, aabb_max = torch.split(self.aabb, 3, dim=-1)
            positions = (positions - aabb_min) / (aabb_max - aabb_min)
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1).to(positions)
        positions = positions * selector.unsqueeze(-1)
        xyz_encoding = self.xyz_encoder(positions.view(-1, self.num_dims))
        density_before_activation = self.base_mlp(xyz_encoding).view(
            list(positions.shape[:-1]) + [-1]
        )
        density = self.density_activation(density_before_activation)
        return {"density": density}


def temporal_interpolation(
    normed_timestamps: Tensor,
    training_timesteps: Tensor,
    normed_positions: Tensor,
    hash_encoder: HashEncoder,
    mlp: nn.Module,
    interpolate_xyz_encoding: bool = False,
) -> Tensor:
    # to be studied
    if len(normed_timestamps.shape) == 2:
        timestep_slice = normed_timestamps[:, 0]
    else:
        timestep_slice = normed_timestamps[:, 0, 0]
    closest_timesteps = find_topk_nearby_timesteps(training_timesteps, timestep_slice)
    if torch.allclose(closest_timesteps[:, 0], timestep_slice):
        temporal_positions = torch.cat([normed_positions, normed_timestamps], dim=-1)
        xyz_encoding = hash_encoder(
            temporal_positions.view(-1, temporal_positions.shape[-1])
        ).view(list(temporal_positions.shape[:-1]) + [-1])
        encoded_feats = mlp(xyz_encoding)
    else:
        left_timesteps, right_timesteps = (
            closest_timesteps[:, 0],
            closest_timesteps[:, 1],
        )
        left_timesteps = left_timesteps.unsqueeze(-1).repeat(
            1, normed_positions.shape[1]
        )
        right_timesteps = right_timesteps.unsqueeze(-1).repeat(
            1, normed_positions.shape[1]
        )
        left_temporal_positions = torch.cat(
            [normed_positions, left_timesteps.unsqueeze(-1)], dim=-1
        )
        right_temporal_positions = torch.cat(
            [normed_positions, right_timesteps.unsqueeze(-1)], dim=-1
        )
        offset = (
            (
                (timestep_slice - left_timesteps[:, 0])
                / (right_timesteps[:, 0] - left_timesteps[:, 0])
            )
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
        left_xyz_encoding = hash_encoder(
            left_temporal_positions.view(-1, left_temporal_positions.shape[-1])
        ).view(list(left_temporal_positions.shape[:-1]) + [-1])
        right_xyz_encoding = hash_encoder(
            right_temporal_positions.view(-1, left_temporal_positions.shape[-1])
        ).view(list(right_temporal_positions.shape[:-1]) + [-1])
        if interpolate_xyz_encoding:
            encoded_feats = mlp(
                left_xyz_encoding * (1 - offset) + right_xyz_encoding * offset
            )
        else:
            encoded_feats = (
                mlp(left_xyz_encoding) * (1 - offset) + mlp(right_xyz_encoding) * offset
            )

    return encoded_feats


def build_radiance_field_from_cfg(cfg, verbose=True) -> RadianceField:
    xyz_encoder = build_xyz_encoder_from_cfg(cfg.xyz_encoder, verbose=verbose)
    dynamic_xyz_encoder = None
    flow_xyz_encoder = None
    if cfg.head.enable_dynamic_branch:
        dynamic_xyz_encoder = build_xyz_encoder_from_cfg(
            cfg.dynamic_xyz_encoder, verbose=verbose
        )
    if cfg.head.enable_flow_branch:
        flow_xyz_encoder = HashEncoder(
            n_input_dims=4,
            n_levels=10,
            base_resolution=16,
            max_resolution=4096,
            log2_hashmap_size=18,
            n_features_per_level=4,
        )
    return RadianceField(
        xyz_encoder=xyz_encoder,
        dynamic_xyz_encoder=dynamic_xyz_encoder,
        flow_xyz_encoder=flow_xyz_encoder,
        unbounded=cfg.unbounded,
        contract_method=cfg.contract_method,
        inner_range=cfg.inner_range,
        contract_ratio=cfg.contract_ratio,
        num_cams=cfg.num_cams,
        geometry_feature_dim=cfg.neck.geometry_feature_dim,
        base_mlp_layer_width=cfg.neck.base_mlp_layer_width,
        head_mlp_layer_width=cfg.head.head_mlp_layer_width,
        enable_segmentation_head=cfg.head.enable_segmentation_head,
        split_semantic_instance=cfg.head.split_semantic_instance,
        segmentation_feature_dim=cfg.neck.segmentation_feature_dim,
        semantic_hidden_dim=cfg.head.semantic_hidden_dim,
        instance_hidden_dim=cfg.head.instance_hidden_dim,
        selection_scale_dim= cfg.head.selection_scale_dim,
        semantic_embedding_dim=cfg.head.semantic_embedding_dim,
        instance_embedding_dim=cfg.head.instance_embedding_dim,
        momentum=cfg.head.momentum,
        enable_cam_embedding=cfg.head.enable_cam_embedding,
        enable_img_embedding=cfg.head.enable_img_embedding,
        appearance_embedding_dim=cfg.head.appearance_embedding_dim,
        enable_sky_head=cfg.head.enable_sky_head,
        enable_shadow_head=cfg.head.enable_shadow_head,
        num_train_timesteps=cfg.num_train_timesteps,  # placeholder
        interpolate_xyz_encoding=cfg.head.interpolate_xyz_encoding,
        enable_temporal_interpolation=cfg.head.enable_temporal_interpolation,
    )


def build_density_field(
    aabb: Union[Tensor, List[float]] = [[-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]],
    type: Literal["HashEncoder"] = "HashEncoder",
    n_input_dims: int = 3,
    n_levels: int = 5,
    base_resolution: int = 16,
    max_resolution: int = 128,
    log2_hashmap_size: int = 20,
    n_features_per_level: int = 2,
    unbounded: bool = True,
    contract_method: str = "aabb_bounded",
    inner_range: List[float] = [50.0, 50.0, 10.0],
    contract_ratio: float = 0.5
) -> DensityField:
    if type == "HashEncoder":
        xyz_encoder = HashEncoder(
            n_input_dims=n_input_dims,
            n_levels=n_levels,
            base_resolution=base_resolution,
            max_resolution=max_resolution,
            log2_hashmap_size=log2_hashmap_size,
            n_features_per_level=n_features_per_level,
        )
    else:
        raise NotImplementedError(f"Unknown (xyz_encoder) type: {type}")
    return DensityField(
        xyz_encoder=xyz_encoder,
        aabb=aabb,
        unbounded=unbounded,
        contract_method=contract_method,
        inner_range=inner_range,
        contract_ratio=contract_ratio
    )


def compute_SRMR(vis_feature: Tensor, clip_text_features: Tensor, sam2_masks: Tensor) -> Tensor:
    """
    Compute the SRMR (Semantic Relevancy Map Refinement) for a given batch feature and
    its related scene classes features.
    :param vis_feature: visual features of the batch (num_rays, D).
    :param clip_text_features: CLIP text features of the scene classes.
    :param sam2_masks: SAM2 masks for the batch.
    :return: A tensor of shape (num_rays,) with the refined relavancy values for each pixel.
    """
    device = vis_feature.device
    
    # Get Clip features 
    # vis_feature = vis_feature.reshape(-1, vis_feature.size(-1)) # [N1, D], N1 is H*W, D is the feature dimension
    clip_text_features_normalized = F.normalize(clip_text_features, dim=1) # [N2, D], N2 is the number of scene classes
    vis_feature_normalized = F.normalize(vis_feature, dim=1) # [N1, D], N1 is num_rays, D is the feature dimension
    # Compute cosine similarity
    relevancy_map = torch.mm(vis_feature_normalized, clip_text_features_normalized.T) # [N1,N2]        
    p_class = F.softmax(relevancy_map, dim=1) # [N1,N2]
    class_index = torch.argmax(p_class, dim=-1) # [N1]
    # pred_index = class_index.reshape(H, W).unsqueeze(0) # [1,H,W]
    pred_index = class_index.unsqueeze(0) # [1,N1]

    # Refine SAM2 masks using the predicted class_index  
    sam_refined_pred = torch.zeros((pred_index.shape[1]), dtype=torch.long).to(device)

    for ann in sam2_masks:
        cur_mask = ann.squeeze()  # [num_rays,], current mask for the annotation                  
        sub_mask = pred_index.squeeze().clone()
        sub_mask[~cur_mask] = 0
        # .view(-1) collapses all dimensions into a single dimension, It is equivalent to tensor.reshape(-1).
        flat_sub_mask = sub_mask.clone().view(-1)           
        flat_sub_mask = flat_sub_mask[flat_sub_mask!=0]
        
        if len(flat_sub_mask) != 0:                         
            unique_elements, counts = torch.unique(flat_sub_mask, return_counts=True)  
            most_common_element = unique_elements[int(counts.argmax().item())]  
        else:                                               
            continue 

        sam_refined_pred[cur_mask] = most_common_element  
    
    logger.info(f"SRMR: {sam_refined_pred.size()}, {sam_refined_pred.device}")
    
    return sam_refined_pred
