# Author: Qian Liu
# Email: liu.qian.pro@gmail.com

import argparse
import json
import logging
import os
import time
from typing import List, Optional

import imageio
import numpy as np
import torch
import torch.utils.data
from omegaconf import OmegaConf
from tqdm import tqdm

import builders
import loss
import utils.misc as misc
import wandb
from datasets import metrics
from datasets.base import SceneDataset
from radiance_fields import DensityField, RadianceField
from radiance_fields.render_utils import render_rays
from radiance_fields.video_utils import render_pixels, save_videos
from third_party.nerfacc_prop_net import PropNetEstimator, get_proposal_requires_grad_fn
from utils.logging import MetricLogger, setup_logging
from utils.visualization_tools import visualize_scene_flow

logger = logging.getLogger()
current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

# a global list of keys to render,
# comment out the keys you don't want to render or uncomment the keys you want to render
render_keys = [
    "gt_rgbs",
    "rgbs",
    "depths",
    "median_depths",
    "dynamic_rgbs",
    "dynamic_depths",
    "static_rgbs",
    "static_depths",
    "forward_flows",
    "backward_flows",
    "shadow_reduced_static_rgbs",
    "shadow_only_static_rgbs",
    "shadows",
    "gt_sky_masks",
    "sky_masks",
]


def get_args_parser():
    parser = argparse.ArgumentParser("Train EmernNerf for a single scene")
    parser.add_argument("--config_file", help="path to config file", type=str)
    parser.add_argument(
        "--eval_only", action="store_true", help="perform evaluation only"
    )
    parser.add_argument(
        "--render_data_video",
        action="store_true",
        help="Render a data video",
    )
    parser.add_argument(
        "--render_data_video_only",
        action="store_true",
        help="Quit after rendering a data video",
    )
    parser.add_argument(
        "--render_video_postfix",
        type=str,
        default=None,
        help="an optional postfix for video",
    )
    parser.add_argument(
        "--output_root",
        default="./work_dirs/",
        help="path to save checkpoints and logs",
        type=str,
    )
    # wandb logging part
    parser.add_argument(
        "--enable_wandb", action="store_true", help="enable wandb logging"
    )
    parser.add_argument(
        "--entity",
        default="i-SLNeRF",
        type=str,
        help="wandb entity name",
        required=False,
    )
    parser.add_argument(
        "--project",
        default="i-SLNeRF",
        type=str,
        help="wandb project name, also used to enhance log_dir",
        required=True,
    )
    parser.add_argument(
        "--run_name",
        default="debug",
        type=str,
        help="wandb run name, also used to enhance log_dir",
        required=True,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def setup(args):
    # ------ get config from args -------- #
    default_config = OmegaConf.create(OmegaConf.load("configs/default_config.yaml"))
    cfg = OmegaConf.load(args.config_file)
    cfg = OmegaConf.merge(default_config, cfg, OmegaConf.from_cli(args.opts))
    log_dir = os.path.join(args.output_root, args.project, args.run_name)
    cfg.log_dir = log_dir
    cfg.nerf.model.num_cams = cfg.data.pixel_source.num_cams
    cfg.nerf.model.unbounded = cfg.nerf.unbounded
    cfg.nerf.model.contract_method = cfg.nerf.contract_method
    cfg.nerf.model.inner_range = cfg.nerf.inner_range
    cfg.nerf.model.contract_ratio = cfg.nerf.contract_ratio
    cfg.nerf.propnet.unbounded = cfg.nerf.unbounded
    cfg.nerf.propnet.contract_method = cfg.nerf.contract_method
    cfg.nerf.propnet.inner_range = cfg.nerf.inner_range
    cfg.nerf.propnet.contract_ratio = cfg.nerf.contract_ratio
    cfg.nerf.model.resume_from = cfg.resume_from
    os.makedirs(log_dir, exist_ok=True)
    for folder in [
        "images",
        "full_videos",
        "test_videos",
        "lowres_videos",
        "metrics",
        "configs_bk",
        "buffer_maps",
    ]:
        os.makedirs(os.path.join(log_dir, folder), exist_ok=True)
    # ------ setup logging -------- #
    if args.enable_wandb:
        # sometimes wandb fails to init in cloud machines, so we give it several (many) tries
        while (
            wandb.init(
                project=args.project,
                entity=args.entity,
                sync_tensorboard=True,
                settings=wandb.Settings(start_method="fork"),
            )
            is not wandb.run
        ):
            continue
        wandb.run.name = args.run_name
        wandb.run.save()
        wandb.config.update(OmegaConf.to_container(cfg, resolve=True))
        wandb.config.update(args)

    misc.fix_random_seeds(cfg.optim.seed)

    global logger
    setup_logging(output=log_dir, level=logging.INFO, time_string=current_time)
    logger.info(
        "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items()))
    )
    # -------- write config -------- #
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    saved_cfg_path = os.path.join(log_dir, "config.yaml")
    with open(saved_cfg_path, "w") as f:
        OmegaConf.save(config=cfg, f=f)
    # also save a backup copy
    saved_cfg_path_bk = os.path.join(
        log_dir, "configs_bk", f"config_{current_time}.yaml"
    )
    with open(saved_cfg_path_bk, "w") as f:
        OmegaConf.save(config=cfg, f=f)
    logger.info(f"Full config saved to {saved_cfg_path}, and {saved_cfg_path_bk}")
    return cfg


@torch.no_grad()
def do_evaluation(
    step: int = 0,
    cfg: OmegaConf = None,
    model: RadianceField = None,
    proposal_networks: Optional[List[DensityField]] = None,
    proposal_estimator: PropNetEstimator = None,
    dataset: SceneDataset = None,
    args: argparse.Namespace = None,
):
    logger.info("Evaluating on the full set...")
    model.eval()
    proposal_estimator.eval()
    for p in proposal_networks:
        p.eval()

    if cfg.eval.eval_lidar_flow and cfg.nerf.model.head.enable_flow_branch:
        assert cfg.data.dataset == "waymo", "only support waymo dataset for now"
        logger.info("Evaluating Lidar Flow...")
        # use metrics from NSFP
        all_flow_metrics = {
            "EPE3D": [],
            "acc3d_strict": [],
            "acc3d_relax": [],
            "angle_error": [],
            "outlier": [],
        }
        for data_dict in tqdm(
            dataset.full_lidar_set, "Evaluating Lidar Flow", dynamic_ncols=True
        ):
            lidar_flow_class = data_dict["lidar_flow_class"]
            for k, v in data_dict.items():
                # remove invalid flow (the information is from GT)
                data_dict[k] = v[lidar_flow_class != -1]

            if data_dict[k].shape[0] == 0:
                logger.info(f"no valid points, skipping...")
                continue

            if cfg.eval.remove_ground_when_eval_lidar_flow:
                # following the setting in scene flow estimation works
                for k, v in data_dict.items():
                    data_dict[k] = v[~data_dict["lidar_ground"]]
            lidar_points = (
                data_dict["lidar_origins"]
                + data_dict["lidar_ranges"] * data_dict["lidar_viewdirs"]
            )
            normalized_timestamps = data_dict["lidar_normed_timestamps"]
            pred_results = model.query_flow(
                positions=lidar_points,
                normed_timestamps=normalized_timestamps,
            )
            pred_flow = pred_results["forward_flow"]
            # flow is only valid when the point is not static
            pred_flow[pred_results["dynamic_density"] < 0.2] *= 0
            # metrics in NSFP
            flow_metrics = metrics.compute_scene_flow_metrics(
                pred_flow[None, ...], data_dict["lidar_flow"][None, ...]
            )
            for k, v in flow_metrics.items():
                all_flow_metrics[k].append(v)
        logger.info("Lidar Flow Results:")
        avg_flow_metrics = {k: np.mean(v) for k, v in all_flow_metrics.items()}
        logger.info(json.dumps(avg_flow_metrics, indent=4))
        flow_metrics_file = f"{cfg.log_dir}/metrics/flow_eval_{current_time}.json"
        with open(flow_metrics_file, "w") as f:
            json.dump(avg_flow_metrics, f)
        logger.info(f"Flow estimation evaluation metrics saved to {flow_metrics_file}")
        if args.enable_wandb:
            wandb.log(avg_flow_metrics)
        torch.cuda.empty_cache()

    if cfg.data.pixel_source.load_rgb and cfg.render.render_low_res:
        logger.info("Rendering full set but in a low_resolution...")
        dataset.pixel_source.update_downscale_factor(1 / cfg.render.low_res_downscale)
        render_results = render_pixels(
            cfg=cfg,
            model=model,
            proposal_networks=proposal_networks,
            proposal_estimator=proposal_estimator,
            dataset=dataset.full_pixel_set,
            compute_metrics=True,
            return_decomposition=True,
        )
        dataset.pixel_source.reset_downscale_factor()
        if args.render_video_postfix is None:
            video_output_pth = os.path.join(cfg.log_dir, "lowres_videos", f"{step}.mp4")
        else:
            video_output_pth = os.path.join(
                cfg.log_dir,
                "lowres_videos",
                f"{step}_{args.render_video_postfix}.mp4",
            )
        vis_frame_dict = save_videos(
            render_results,
            video_output_pth,
            num_timestamps=dataset.num_img_timesteps,
            keys=render_keys,
            save_seperate_video=cfg.logging.save_seperate_video,
            num_cams=dataset.pixel_source.num_cams,
            fps=cfg.render.fps,
            verbose=True,
        )
        if args.enable_wandb:
            for k, v in vis_frame_dict.items():
                wandb.log({f"pixel_rendering/lowres_full/{k}": wandb.Image(v)})

        del render_results, vis_frame_dict
        torch.cuda.empty_cache()

    if cfg.data.pixel_source.load_rgb:
        logger.info("Evaluating Pixels...")
        if dataset.test_pixel_set is not None and cfg.render.render_test:
            logger.info("Evaluating Test Set Pixels...")
            render_results = render_pixels(
                cfg=cfg,
                model=model,
                proposal_estimator=proposal_estimator,
                dataset=dataset.test_pixel_set,
                proposal_networks=proposal_networks,
                compute_metrics=True,
                return_decomposition=True,
            )
            eval_dict = {}
            for k, v in render_results.items():
                if k in [
                    "psnr",
                    "ssim",
                    "depth_rmse"
                ]:
                    eval_dict[f"pixel_metrics/test/{k}"] = v
            if args.enable_wandb:
                wandb.log(eval_dict)
            test_metrics_file = f"{cfg.log_dir}/metrics/images_test_{current_time}.json"
            with open(test_metrics_file, "w") as f:
                json.dump(eval_dict, f)
            logger.info(f"Image evaluation metrics saved to {test_metrics_file}")

            if args.render_video_postfix is None:
                video_output_pth = f"{cfg.log_dir}/test_videos/{step}.mp4"
            else:
                video_output_pth = (
                    f"{cfg.log_dir}/test_videos/{step}_{args.render_video_postfix}.mp4"
                )
            vis_frame_dict = save_videos(
                render_results,
                video_output_pth,
                num_timestamps=dataset.num_test_timesteps,
                keys=render_keys,
                num_cams=dataset.pixel_source.num_cams,
                save_seperate_video=cfg.logging.save_seperate_video,
                fps=cfg.render.fps,
                verbose=True,
                # save_images=True,
            )
            if args.enable_wandb:
                for k, v in vis_frame_dict.items():
                    wandb.log({"pixel_rendering/test/" + k: wandb.Image(v)})
            del render_results, vis_frame_dict
            torch.cuda.empty_cache()
        if cfg.render.render_full:
            logger.info("Evaluating Full Set...")
            render_results = render_pixels(
                cfg=cfg,
                model=model,
                proposal_estimator=proposal_estimator,
                dataset=dataset.full_pixel_set,
                proposal_networks=proposal_networks,
                compute_metrics=True,
                return_decomposition=True,
            )
            eval_dict = {}
            for k, v in render_results.items():
                if k in [
                    "psnr",
                    "ssim",
                    "depth_rmse"
                ]:
                    eval_dict[f"pixel_metrics/full/{k}"] = v
            if args.enable_wandb:
                wandb.log(eval_dict)
            test_metrics_file = f"{cfg.log_dir}/metrics/images_full_{current_time}.json"
            with open(test_metrics_file, "w") as f:
                json.dump(eval_dict, f)
            logger.info(f"Image evaluation metrics saved to {test_metrics_file}")

            if args.render_video_postfix is None:
                video_output_pth = f"{cfg.log_dir}/full_videos/{step}.mp4"
            else:
                video_output_pth = (
                    f"{cfg.log_dir}/full_videos/{step}_{args.render_video_postfix}.mp4"
                )
            vis_frame_dict = save_videos(
                render_results,
                video_output_pth,
                num_timestamps=dataset.num_img_timesteps,
                keys=render_keys,
                num_cams=dataset.pixel_source.num_cams,
                save_seperate_video=cfg.logging.save_seperate_video,
                fps=cfg.render.fps,
                verbose=True,
            )
            if args.enable_wandb:
                for k, v in vis_frame_dict.items():
                    wandb.log({"pixel_rendering/full/" + k: wandb.Image(v)})
            del render_results, vis_frame_dict
            torch.cuda.empty_cache()


######################################TRAINING######################################
def assemble_dataset(cfg):
    if cfg.data.dataset == "waymo":
        from datasets.waymo import WaymoDataset

        dataset = WaymoDataset(data_cfg=cfg.data)
    else:
        from datasets.nuscenes import NuScenesDataset

        dataset = NuScenesDataset(data_cfg=cfg.data)

    # To give us a quick preview of the scene, we render a data video
    if args.render_data_video or args.render_data_video_only:
        save_pth = os.path.join(cfg.log_dir, "data.mp4")
        # define a `render_data_videos` per dataset.
        dataset.render_data_videos(save_pth=save_pth, split="full")
        if args.render_data_video_only:
            logger.info("Render data video only, exiting...")
            exit()
    
    return dataset


def construct_training_objects(cfg, dataset, device):
    (
        proposal_estimator,
        proposal_networks,
    ) = builders.build_estimator_and_propnet_from_cfg(
        nerf_cfg=cfg.nerf, optim_cfg=cfg.optim, dataset=dataset, device=device
    )
    model = builders.build_model_from_cfg(
        cfg=cfg.nerf.model, dataset=dataset, device=device
    )
    logger.info(f"PropNetEstimator: {proposal_networks}")
    logger.info(f"Model: {model}")

    # ------ build optimizer and grad scaler -------- #
    optimizer = builders.build_optimizer_from_cfg(cfg=cfg.optim, model=model)
    pixel_grad_scaler = torch.cuda.amp.GradScaler(2**10)
    lidar_grad_scaler = torch.cuda.amp.GradScaler(2**10)
    semantic_grad_scaler = torch.cuda.amp.GradScaler(2**10)
    instance_grad_scaler = torch.cuda.amp.GradScaler(2**10)

    # ------ build scheduler -------- #
    scheduler = builders.build_scheduler_from_cfg(cfg=cfg.optim, optimizer=optimizer)

    if cfg.resume_from is not None:
        start_step = misc.resume_from_checkpoint(
            ckpt_path=cfg.resume_from,
            model=model,
            proposal_networks=proposal_networks,
            proposal_estimator=proposal_estimator,
            optimizer=optimizer,
            scheduler=scheduler,
        )
    else:
        start_step = 0
        logger.info(
            f"Will start training for {cfg.optim.num_iters} iterations from scratch"
        )
    
    return model, proposal_estimator, proposal_networks, optimizer, \
        scheduler, start_step, pixel_grad_scaler, lidar_grad_scaler, semantic_grad_scaler,\
        instance_grad_scaler


def build_static_losses(cfg):
    # rgb loss
    if cfg.data.pixel_source.load_rgb:
        rgb_loss_fn = loss.RealValueLoss(
            loss_type=cfg.supervision.rgb.loss_type,
            coef=cfg.supervision.rgb.loss_coef,
            name="rgb",
            check_nan=cfg.optim.check_nan,
        )

    # lidar related losses
    if cfg.data.lidar_source.load_lidar and cfg.supervision.depth.enable:
        depth_loss_fn = loss.DepthLoss(
            loss_type=cfg.supervision.depth.loss_type,
            coef=cfg.supervision.depth.loss_coef,
            depth_error_percentile=cfg.supervision.depth.depth_error_percentile,
            check_nan=cfg.optim.check_nan,
        )
        if cfg.supervision.depth.line_of_sight.enable:
            line_of_sight_loss_fn = loss.LineOfSightLoss(
                loss_type=cfg.supervision.depth.line_of_sight.loss_type,
                name="line_of_sight",
                depth_error_percentile=cfg.supervision.depth.depth_error_percentile,
                coef=cfg.supervision.depth.line_of_sight.loss_coef,
                check_nan=cfg.optim.check_nan,
            )
        else:
            line_of_sight_loss_fn = None
    elif not cfg.data.lidar_source.load_lidar and cfg.data.pixel_source.load_depth_map:
        depth_loss_fn = loss.VisionDepthLoss(
            loss_type=cfg.supervision.vision_depth.loss_type,
            coef=cfg.supervision.vision_depth.loss_coef,
            max_depth=cfg.supervision.vision_depth.max_depth
        )
        line_of_sight_loss_fn = None
    else:
        depth_loss_fn = None
        line_of_sight_loss_fn = None

    if cfg.data.pixel_source.load_sky_mask and cfg.nerf.model.head.enable_sky_head:
        sky_loss_fn = loss.SkyLoss(
            loss_type=cfg.supervision.sky.loss_type,
            coef=cfg.supervision.sky.loss_coef,
            check_nan=cfg.optim.check_nan,
        )
    else:
        sky_loss_fn = None

    return rgb_loss_fn, depth_loss_fn, line_of_sight_loss_fn, sky_loss_fn


def build_dynamic_losses(cfg):
    ## ------ dynamic related losses -------- #
    if cfg.nerf.model.head.enable_dynamic_branch:
        dynamic_reg_loss_fn = loss.DynamicRegularizationLoss(
            loss_type=cfg.supervision.dynamic.loss_type,
            coef=cfg.supervision.dynamic.loss_coef,
            entropy_skewness=cfg.supervision.dynamic.entropy_loss_skewness,
            check_nan=cfg.optim.check_nan,
        )
    else:
        dynamic_reg_loss_fn = None

    if cfg.nerf.model.head.enable_shadow_head:
        shadow_loss_fn = loss.DynamicRegularizationLoss(
            name="shadow",
            loss_type=cfg.supervision.shadow.loss_type,
            coef=cfg.supervision.shadow.loss_coef,
            check_nan=cfg.optim.check_nan,
        )
    else:
        shadow_loss_fn = None
    
    return dynamic_reg_loss_fn, shadow_loss_fn


def buid_segmentation_losses(cfg):
    if cfg.data.pixel_source.load_segmentation and cfg.nerf.model.head.enable_segmentation_head:
        semantic_classfication_loss_fn = loss.Semantic_Classification_Loss(
            coef=cfg.supervision.segmentation.semantic.loss_coef
        )
        instance_consistency_loss_fn = loss.Instance_Consistency_Loss(
            coef=cfg.supervision.segmentation.instance.loss_coef
        )
    else:
        semantic_classfication_loss_fn = None
        instance_consistency_loss_fn = None
    
    return semantic_classfication_loss_fn, instance_consistency_loss_fn


def compute_pixel_losses(cfg, step, dataset, model, proposal_estimator, proposal_networks,
                         proposal_requires_grad_fn, pixel_loss_dict, rgb_loss_fn, sky_loss_fn,
                         depth_loss_fn, dynamic_reg_loss_fn, shadow_loss_fn, optimizer,
                         pixel_grad_scaler, scheduler):
    # ------ pixel ray supervision -------- #
    if cfg.data.pixel_source.load_rgb:
        proposal_requires_grad = proposal_requires_grad_fn(int(step))
        i = torch.randint(0, len(dataset.train_pixel_set), (1,)).item()
        pixel_data_dict = dataset.train_pixel_set[i]
        for k, v in pixel_data_dict.items():
            if isinstance(v, torch.Tensor):
                pixel_data_dict[k] = v.cuda(non_blocking=True)
        
        # ------ pixel-wise supervision -------- #
        render_results = render_rays(
            radiance_field=model,
            proposal_estimator=proposal_estimator,
            proposal_networks=proposal_networks,
            data_dict=pixel_data_dict,
            cfg=cfg,
            proposal_requires_grad=proposal_requires_grad,
        )
        proposal_estimator.update_every_n_steps(
            render_results["extras"]["trans"],
            proposal_requires_grad,
            loss_scaler=1024,
        )

        # compute losses
        # rgb loss
        pixel_loss_dict.update(
            rgb_loss_fn(render_results["rgb"], pixel_data_dict["pixels"])
        )
        if sky_loss_fn is not None:  # if sky loss is enabled
            if cfg.supervision.sky.loss_type == "weights_based":
                # penalize the points' weights if they point to the sky
                pixel_loss_dict.update(
                    sky_loss_fn(
                        render_results["extras"]["weights"],
                        pixel_data_dict["sky_masks"],
                    )
                )
            elif cfg.supervision.sky.loss_type == "opacity_based":
                # penalize accumulated opacity if the ray points to the sky
                pixel_loss_dict.update(
                    sky_loss_fn(
                        render_results["opacity"], pixel_data_dict["sky_masks"]
                    )
                )
            else:
                raise NotImplementedError(
                    f"sky_loss_type {cfg.supervision.sky.loss_type} not implemented"
                )
        # vision depth loss
        if depth_loss_fn is not None and not cfg.data.lidar_source.load_lidar:
            pixel_loss_dict.update(
                depth_loss_fn(
                    render_results["depth"],
                    pixel_data_dict["depth_maps"]
                )
            )
        # dynamic and shadow loss
        if dynamic_reg_loss_fn is not None:
            pixel_loss_dict.update(
                dynamic_reg_loss_fn(
                    dynamic_density=render_results["extras"]["dynamic_density"],
                    static_density=render_results["extras"]["static_density"],
                )
            )
        if shadow_loss_fn is not None:
            render_results["shadow_ratio"].nan_to_num_(nan=1e-6, posinf=1.0, neginf=1.0)
            pixel_loss_dict.update(
                shadow_loss_fn(
                    render_results["shadow_ratio"],
                )
            )
        # cyclic flow loss
        if "forward_flow" in render_results["extras"]:
            render_results["extras"]["forward_flow"].nan_to_num_(nan=0.0, posinf=1.0, neginf=1.0)
            render_results["extras"]["backward_flow"].nan_to_num_(nan=0.0, posinf=1.0, neginf=1.0)
            render_results["extras"]["forward_pred_backward_flow"].nan_to_num_(nan=0.0, posinf=1.0, neginf=1.0)
            render_results["extras"]["backward_pred_forward_flow"].nan_to_num_(nan=0.0, posinf=1.0, neginf=1.0)
            cycle_loss = (
                0.5
                * (
                    (
                        render_results["extras"]["forward_flow"].detach()
                        + render_results["extras"]["forward_pred_backward_flow"]
                    )
                    ** 2
                    + (
                        render_results["extras"]["backward_flow"].detach()
                        + render_results["extras"]["backward_pred_forward_flow"]
                    )
                    ** 2
                ).mean()
            )
            pixel_loss_dict.update({"cycle_loss": cycle_loss * 0.01})
            stats = {
                "max_forward_flow_norm": (
                    render_results["extras"]["forward_flow"]
                    .detach()
                    .norm(dim=-1)
                    .max()
                ),
                "max_backward_flow_norm": (
                    render_results["extras"]["backward_flow"]
                    .detach()
                    .norm(dim=-1)
                    .max()
                ),
                "max_forward_pred_backward_flow_norm": (
                    render_results["extras"]["forward_pred_backward_flow"]
                    .norm(dim=-1)
                    .max()
                ),
                "max_backward_pred_forward_flow_norm": (
                    render_results["extras"]["backward_pred_forward_flow"]
                    .norm(dim=-1)
                    .max()
                ),
            }
        total_pixel_loss = sum(loss for loss in pixel_loss_dict.values())
        optimizer.zero_grad()
        pixel_grad_scaler.scale(total_pixel_loss).backward()
        optimizer.step()
        scheduler.step()
    
    return stats, total_pixel_loss, pixel_data_dict, render_results


def compute_lidar_losses(cfg, step, dataset, model, proposal_estimator, proposal_networks,
                         proposal_requires_grad_fn, lidar_loss_dict, depth_loss_fn, line_of_sight_loss_fn,
                         dynamic_reg_loss_fn, lidar_grad_scaler, optimizer, scheduler, epsilon_start,
                         epsilon_final, line_of_sight_loss_decay_weight):
    # ------ lidar ray supervision -------- #
    if cfg.data.lidar_source.load_lidar and cfg.supervision.depth.enable:
        proposal_requires_grad = proposal_requires_grad_fn(int(step))
        i = torch.randint(0, len(dataset.train_lidar_set), (1,)).item()
        lidar_data_dict = dataset.train_lidar_set[i]
        for k, v in lidar_data_dict.items():
            if isinstance(v, torch.Tensor):
                lidar_data_dict[k] = v.cuda(non_blocking=True)
        lidar_render_results = render_rays(
            radiance_field=model,
            proposal_estimator=proposal_estimator,
            proposal_networks=proposal_networks,
            data_dict=lidar_data_dict,
            cfg=cfg,
            proposal_requires_grad=proposal_requires_grad,
            prefix="lidar_",
        )
        proposal_estimator.update_every_n_steps(
            lidar_render_results["extras"]["trans"],
            proposal_requires_grad,
            loss_scaler=1024,
        )
        lidar_loss_dict.update(
            depth_loss_fn(
                lidar_render_results["depth"],
                lidar_data_dict["lidar_ranges"],
                name="lidar_range_loss",
            )
        )
        epsilon = None
        if (
            line_of_sight_loss_fn is not None
            and step > cfg.supervision.depth.line_of_sight.start_iter
        ):
            m = (epsilon_final - epsilon_start) / (
                cfg.optim.num_iters - cfg.supervision.depth.line_of_sight.start_iter
            )
            b = epsilon_start - m * cfg.supervision.depth.line_of_sight.start_iter

            def epsilon_decay(step):
                if step < cfg.supervision.depth.line_of_sight.start_iter:
                    return epsilon_start
                elif step > cfg.optim.num_iters:
                    return epsilon_final
                else:
                    return m * step + b

            epsilon = epsilon_decay(step)
            line_of_sight_loss_dict = line_of_sight_loss_fn(
                pred_depth=lidar_render_results["depth"],
                gt_depth=lidar_data_dict["lidar_ranges"],
                weights=lidar_render_results["extras"]["weights"],
                t_vals=lidar_render_results["extras"]["t_vals"],
                epsilon=epsilon,
                name="lidar_line_of_sight",
                coef_decay=line_of_sight_loss_decay_weight,
            )
            lidar_loss_dict.update(
                {
                    "lidar_line_of_sight": line_of_sight_loss_dict[
                        "lidar_line_of_sight"
                    ].mean()
                }
            )

        if dynamic_reg_loss_fn is not None:
            lidar_loss_dict.update(
                dynamic_reg_loss_fn(
                    dynamic_density=lidar_render_results["extras"][
                        "dynamic_density"
                    ],
                    static_density=lidar_render_results["extras"]["static_density"],
                    name="lidar_dynamic",
                )
            )

        total_lidar_loss = sum(loss for loss in lidar_loss_dict.values())
        optimizer.zero_grad()
        lidar_grad_scaler.scale(total_lidar_loss).backward()
        optimizer.step()
        scheduler.step()
        total_lidar_loss = total_lidar_loss.item()
    else:
        total_lidar_loss = -1
        lidar_data_dict = None
        lidar_render_results = None
        epsilon = None
    
    return total_lidar_loss, lidar_data_dict, lidar_render_results, epsilon


def compute_segmentation_loss(cfg, step, dataset, model, proposal_estimator, proposal_networks,
                              proposal_requires_grad_fn, semantic_loss_dict, instance_loss_dict,
                              semantic_classfication_loss_fn, instance_consistency_loss_fn, optimizer,
                              semantic_grad_scaler, instance_grad_scaler, scheduler):
    if cfg.data.pixel_source.load_segmentation and cfg.nerf.model.head.enable_segmentation_head and \
        step >= cfg.supervision.segmentation.semantic.start_iter:
        proposal_requires_grad = proposal_requires_grad_fn(int(step))
        i = torch.randint(0, len(dataset.train_pixel_set), (1,)).item()
        seg_pixel_data_dict = dataset.train_pixel_set[i]
        for k, v in seg_pixel_data_dict.items():
            if isinstance(v, torch.Tensor):
                seg_pixel_data_dict[k] = v.cuda(non_blocking=True)
        
        # ------ pixel-wise supervision -------- #
        segmentation_render_results = render_rays(
            radiance_field=model,
            proposal_estimator=proposal_estimator,
            proposal_networks=proposal_networks,
            data_dict=seg_pixel_data_dict,
            cfg=cfg,
            proposal_requires_grad=proposal_requires_grad
        )
        proposal_estimator.update_every_n_steps(
            segmentation_render_results["extras"]["trans"],
            proposal_requires_grad,
            loss_scaler=1024,
        )

        total_semantic_loss = 0
        total_instance_loss = 0

        # compute semantic losses
        semantic_loss_dict.update(semantic_classfication_loss_fn(
            segmentation_render_results["semantic_embedding"],
            seg_pixel_data_dict["semantic_masks"]
            )
        )
        
        total_semantic_loss = sum(loss for loss in semantic_loss_dict.values())
        optimizer.zero_grad()
        semantic_grad_scaler.scale(total_semantic_loss).backward()
        optimizer.step()
        scheduler.step()
        
        # compute instance consistency loss
        if step >= cfg.supervision.segmentation.instance.start_iter:
            model.ema_update_slownet(model.slow_instance_head, model.fast_instance_head)
            instance_loss_dict.update(instance_consistency_loss_fn(
                segmentation_render_results["instance_embedding"],
                seg_pixel_data_dict["instance_masks"],
                seg_pixel_data_dict["instance_confidences"]
                )
            )
            total_instance_loss = sum(loss for loss in instance_loss_dict.values())
            optimizer.zero_grad()
            segmentation_render_results["instance_embedding"].retain_grad()
            instance_grad_scaler.scale(total_instance_loss).backward()
            optimizer.step()
            scheduler.step()

    else:
        total_semantic_loss = -1
        total_instance_loss = -1

    return total_semantic_loss, total_instance_loss   


def log_metrics(metric_logger, pixel_data_dict, lidar_data_dict, render_results, lidar_render_results,
                optimizer, pixel_loss_dict, lidar_loss_dict, semantic_loss_dict, instance_loss_dict,
                total_pixel_loss, total_lidar_loss, total_semantic_loss, total_instance_loss, stats,
                epsilon, args):
    # ------ log metric values -------- #
    if pixel_data_dict is not None:
        psnr = metrics.compute_psnr(
            render_results["rgb"], pixel_data_dict["pixels"]
        )
        metric_logger.update(psnr=psnr)
        metric_logger.update(
            total_pixel_loss=total_pixel_loss.item(),
        )

    if lidar_data_dict is not None:
        metric_logger.update(
            total_lidar_loss=total_lidar_loss,
        )
        range_rmse = metrics.compute_valid_depth_rmse(
            lidar_render_results["depth"], lidar_data_dict["lidar_ranges"]
        )
        metric_logger.update(range_rmse=range_rmse)

    if semantic_loss_dict is not None:
        metric_logger.update(
            total_semantic_loss=total_semantic_loss
        )
    if instance_loss_dict is not None:
        metric_logger.update(
            total_instance_loss=total_instance_loss
        )

    metric_logger.update(**{k: v.item() for k, v in pixel_loss_dict.items()})
    metric_logger.update(**{k: v.item() for k, v in lidar_loss_dict.items()})
    metric_logger.update(**{k: v.item() for k, v in semantic_loss_dict.items()})
    metric_logger.update(**{k: v.item() for k, v in instance_loss_dict.items()})
    metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    if stats is not None:
        metric_logger.update(**{k: v.item() for k, v in stats.items()})
    if epsilon is not None:
        metric_logger.update(epsilon=epsilon)
    # log to wandb
    if args.enable_wandb:
        wandb.log(
            {f"train_stats/{k}": v.avg for k, v in metric_logger.meters.items()}
        )


def save_checkpoint(step, cfg, model, proposal_estimator, proposal_networks, optimizer, scheduler):
    # ------ save checkpoints -------- #
    if step > 0 and (
        ((step % cfg.logging.saveckpt_freq == 0 and step >= 16000) or (step == cfg.optim.num_iters))
        and (cfg.resume_from is None)
    ):
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "proposal_networks": [p.state_dict() for p in proposal_networks],
            "estimator.optimizer": proposal_estimator.optimizer.state_dict(),
            "estimator.scheduler": proposal_estimator.scheduler.state_dict(),
            "step": step,
        }
        save_pth = os.path.join(cfg.log_dir, f"checkpoint_{step:05d}.pth")
        torch.save(checkpoint, save_pth)
        logger.info(f"Saved a checkpoint to {save_pth}")


def eval_rgb_color_map(step, cfg, model, proposal_networks, proposal_estimator, dataset):
    # ------ evaluation, rgb error map -------- #
    if (
        step > 0
        and cfg.data.pixel_source.load_rgb
        and step % cfg.optim.cache_rgb_freq == 0
    ):
        model.eval()
        proposal_estimator.eval()
        for p in proposal_networks:
            p.eval()
        if cfg.data.pixel_source.sampler.buffer_ratio > 0:
            with torch.no_grad():
                logger.info("cache rgb error map...")
                dataset.pixel_source.update_downscale_factor(
                    1 / cfg.data.pixel_source.sampler.buffer_downscale
                )
                render_results = render_pixels(
                    cfg=cfg,
                    model=model,
                    proposal_networks=proposal_networks,
                    proposal_estimator=proposal_estimator,
                    dataset=dataset.full_pixel_set,
                    compute_metrics=False,
                    return_decomposition=True,
                )
                dataset.pixel_source.reset_downscale_factor()
                dataset.pixel_source.update_pixel_error_maps(render_results)
                maps_video = dataset.pixel_source.get_pixel_sample_weights_video()
                merged_list = []
                for i in range(len(maps_video) // dataset.pixel_source.num_cams):
                    frames = maps_video[
                        i
                        * dataset.pixel_source.num_cams : (i + 1)
                        * dataset.pixel_source.num_cams
                    ]
                    frames = [
                        np.stack([frame, frame, frame], axis=-1) for frame in frames
                    ]
                    frames = np.concatenate(frames, axis=1)
                    merged_list.append(frames)
                merged_video = np.stack(merged_list, axis=0)
                merged_video -= merged_video.min()
                merged_video /= merged_video.max()
                merged_video = np.clip(merged_video * 255, 0, 255).astype(np.uint8)

                imageio.mimsave(
                    os.path.join(
                        cfg.log_dir, "buffer_maps", f"buffer_maps_{step}.mp4"
                    ),
                    merged_video,
                    fps=cfg.render.fps,
                )
            logger.info("Done caching rgb error maps")


def visualize_during_training(step, cfg, model, proposal_networks, proposal_estimator, dataset, args):
    # ------ visualization during training-------- #
    if step > 0 and step % cfg.logging.vis_freq == 0:
        model.eval()
        proposal_estimator.eval()
        for p in proposal_networks:
            p.eval()
        if cfg.data.pixel_source.load_rgb:
            logger.info("Visualizing...")
            vis_timestep = np.linspace(
                0,
                dataset.num_img_timesteps,
                cfg.optim.num_iters // cfg.logging.vis_freq + 1,
                endpoint=False,
                dtype=int,
            )[step // cfg.logging.vis_freq]
            with torch.no_grad():
                render_results = render_pixels(
                    cfg=cfg,
                    model=model,
                    proposal_networks=proposal_networks,
                    proposal_estimator=proposal_estimator,
                    dataset=dataset.full_pixel_set,
                    compute_metrics=True,
                    vis_indices=[
                        vis_timestep * dataset.pixel_source.num_cams + i
                        for i in range(dataset.pixel_source.num_cams)
                    ],
                    return_decomposition=True,
                )
            if args.enable_wandb:
                wandb.log(
                    {
                        "pixel_metrics/psnr": render_results["psnr"],
                        "pixel_metrics/ssim": render_results["ssim"],
                        "pixel_metrics/depth_rmse": render_results["depth_rmse"]
                    }
                )
            vis_frame_dict = save_videos(
                render_results,
                save_pth=os.path.join(
                    cfg.log_dir, "images", f"step_{step}.png"
                ),  # don't save the video
                num_timestamps=1,
                keys=render_keys,
                save_seperate_video=cfg.logging.save_seperate_video,
                num_cams=dataset.pixel_source.num_cams,
                fps=cfg.render.fps,
                verbose=False,
            )
            if args.enable_wandb:
                for k, v in vis_frame_dict.items():
                    wandb.log({"pixel_rendering/" + k: wandb.Image(v)})
            if cfg.data.pixel_source.sampler.buffer_ratio > 0:
                vis_frame = dataset.pixel_source.visualize_pixel_sample_weights(
                    [
                        vis_timestep * dataset.pixel_source.num_cams + i
                        for i in range(dataset.pixel_source.num_cams)
                    ]
                )
                imageio.imwrite(
                    os.path.join(
                        cfg.log_dir, "buffer_maps", f"buffer_map_{step}.png"
                    ),
                    vis_frame,
                )
                if args.enable_wandb:
                    wandb.log(
                        {"pixel_rendering/buffer_map": wandb.Image(vis_frame)}
                    )
            del render_results
            torch.cuda.empty_cache()


def start_training_loop(metric_logger, all_iters, cfg, args, model, proposal_estimator, proposal_networks,
                        proposal_requires_grad_fn, optimizer, scheduler, pixel_grad_scaler, lidar_grad_scaler,
                        semantic_grad_scaler, instance_grad_scaler, dataset, rgb_loss_fn, depth_loss_fn,
                        line_of_sight_loss_fn, epsilon_start, epsilon_final, line_of_sight_loss_decay_weight,
                        sky_loss_fn, dynamic_reg_loss_fn, shadow_loss_fn, semantic_classfication_loss_fn,
                        instance_consistency_loss_fn):
    for step in metric_logger.log_every(all_iters, cfg.logging.print_freq):
        model.train()
        proposal_estimator.train()
        for p in proposal_networks:
            p.train()
        
        pixel_loss_dict, lidar_loss_dict, semantic_loss_dict, instance_loss_dict = {}, {}, {}, {}

        stats, total_pixel_loss, pixel_data_dict, render_results =\
                                  compute_pixel_losses(cfg, step, dataset, model, proposal_estimator,
                                                       proposal_networks, proposal_requires_grad_fn,
                                                       pixel_loss_dict, rgb_loss_fn, sky_loss_fn,
                                                       depth_loss_fn, dynamic_reg_loss_fn, shadow_loss_fn,
                                                       optimizer, pixel_grad_scaler, scheduler)

        total_lidar_loss, lidar_data_dict, lidar_render_results, epsilon =\
                           compute_lidar_losses(cfg, step, dataset, model, proposal_estimator,
                                                proposal_networks, proposal_requires_grad_fn,
                                                lidar_loss_dict, depth_loss_fn, line_of_sight_loss_fn,
                                                dynamic_reg_loss_fn, lidar_grad_scaler, optimizer,
                                                scheduler, epsilon_start, epsilon_final,
                                                line_of_sight_loss_decay_weight)

        total_semantic_loss, total_instance_loss =\
            compute_segmentation_loss(cfg, step, dataset, model, proposal_estimator, proposal_networks,
                              proposal_requires_grad_fn, semantic_loss_dict, instance_loss_dict,
                              semantic_classfication_loss_fn, instance_consistency_loss_fn, optimizer,
                              semantic_grad_scaler, instance_grad_scaler, scheduler)

        log_metrics(metric_logger, pixel_data_dict, lidar_data_dict, render_results, lidar_render_results,
                optimizer, pixel_loss_dict, lidar_loss_dict, semantic_loss_dict, instance_loss_dict,
                total_pixel_loss, total_lidar_loss, total_semantic_loss, total_instance_loss, stats,
                epsilon, args)

        save_checkpoint(step, cfg, model, proposal_estimator, proposal_networks, optimizer, scheduler)

        eval_rgb_color_map(step, cfg, model, proposal_networks, proposal_estimator, dataset)

        visualize_during_training(step, cfg, model, proposal_networks, proposal_estimator, dataset, args)

    logger.info("Training done!")

    return step


def main(args):
    cfg = setup(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ------ build dataset -------- #
    # we need to set some hyper-parameters for the model based on the dataset,
    # e.g., aabb, number of training timestamps, number of cameras, etc, so
    # we build the dataset at first.
    dataset = assemble_dataset(cfg)

    # ------ build proposal networks and models -------- #
    # we input the dataset to the model builder to set some hyper-parameters
    model, proposal_estimator, proposal_networks, optimizer, \
        scheduler, start_step, pixel_grad_scaler, lidar_grad_scaler,\
            semantic_grad_scaler, instance_grad_scaler = construct_training_objects(cfg, dataset, device)

    if args.eval_only:
        if cfg.nerf.model.head.enable_flow_branch:
            logger.info("Visualizing scene flow...")
            visualize_scene_flow(
                cfg=cfg,
                model=model,
                dataset=dataset,
                device=device,
            )
        logger.info("Visualization done!")

        do_evaluation(
            step=start_step,
            cfg=cfg,
            model=model,
            proposal_networks=proposal_networks,
            proposal_estimator=proposal_estimator,
            dataset=dataset,
            args=args
        )
        exit()

    # ------ build losses -------- #
    rgb_loss_fn, depth_loss_fn, line_of_sight_loss_fn, sky_loss_fn = build_static_losses(cfg)
    dynamic_reg_loss_fn, shadow_loss_fn = build_dynamic_losses(cfg)
    semantic_classfication_loss_fn, instance_consistency_loss_fn = buid_segmentation_losses(cfg)

    metrics_file = os.path.join(cfg.log_dir, "metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
    proposal_requires_grad_fn = get_proposal_requires_grad_fn()

    epsilon_final = cfg.supervision.depth.line_of_sight.end_epsilon
    epsilon_start = cfg.supervision.depth.line_of_sight.start_epsilon
    all_iters = np.arange(start_step, cfg.optim.num_iters + 1)
    line_of_sight_loss_decay_weight = 1.0

    # ------ start training -------- #
    step = start_training_loop(
        metric_logger, all_iters, cfg, args, model, proposal_estimator, proposal_networks,
        proposal_requires_grad_fn, optimizer, scheduler, pixel_grad_scaler, lidar_grad_scaler,
        semantic_grad_scaler, instance_grad_scaler, dataset, rgb_loss_fn, depth_loss_fn, line_of_sight_loss_fn,
        epsilon_start, epsilon_final, line_of_sight_loss_decay_weight, sky_loss_fn, dynamic_reg_loss_fn,
        shadow_loss_fn, semantic_classfication_loss_fn, instance_consistency_loss_fn)    

    do_evaluation(
        step=step,
        cfg=cfg,
        model=model,
        proposal_networks=proposal_networks,
        proposal_estimator=proposal_estimator,
        dataset=dataset,
        args=args,
    )
    
    if args.enable_wandb:
        wandb.finish()


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
