from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import moviepy as mpy
import torch
import wandb
from einops import pack, rearrange, repeat, einsum
from jaxtyping import Float
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor, nn, optim
import numpy as np
import json
import os
import time
from tqdm import tqdm
import torch.nn.functional as F
import math

from ..dataset.data_module import get_data_shim
from ..dataset.types import BatchedExample
from ..dataset import DatasetCfg
from ..evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from ..global_cfg import get_cfg
from ..loss import Loss
from ..misc.benchmarker import Benchmarker
from ..misc.image_io import prep_image, save_image, save_video
from ..misc.LocalLogger import LOG_PATH, LocalLogger
from ..misc.step_tracker import StepTracker
from ..visualization.annotation import add_label
from ..visualization.camera_trajectory.interpolation import (
    interpolate_extrinsics,
    interpolate_intrinsics,
)
from ..visualization.camera_trajectory.wobble import (
    generate_wobble,
    generate_wobble_transformation,
)
from ..visualization.color_map import apply_color_map_to_image
from ..visualization.layout import add_border, hcat, vcat
from ..visualization.validation_in_3d import render_cameras, render_projections
from .decoder.decoder import Decoder, DepthRenderingMode, DecoderOutput
from .encoder import Encoder
from .encoder.visualization.encoder_visualizer import EncoderVisualizer
from src.visualization.vis_depth import viz_depth_tensor
from PIL import Image
from ..misc.stablize_camera import render_stabilization_path


@dataclass
class OptimizerCfg:
    lr: float
    warm_up_steps: int
    lr_monodepth: float
    weight_decay: float


@dataclass
class TestCfg:
    output_path: Path
    compute_scores: bool
    save_image: bool
    save_video: bool
    eval_time_skip_steps: int
    save_gt_image: bool
    save_input_images: bool
    save_depth: bool
    save_depth_concat_img: bool
    save_depth_npy: bool
    render_chunk_size: int | None
    stablize_camera: bool
    stab_camera_kernel: int


@dataclass
class TrainCfg:
    depth_mode: DepthRenderingMode | None
    extended_visualization: bool
    print_log_every_n_steps: int
    eval_model_every_n_val: int
    eval_data_length: int
    eval_deterministic: bool
    eval_time_skip_steps: int
    eval_save_model: bool
    l1_loss: bool
    intermediate_loss_weight: float
    no_viz_video: bool
    viz_depth: bool
    forward_depth_only: bool
    train_ignore_large_loss: float
    no_log_projections: bool


@runtime_checkable
class TrajectoryFn(Protocol):
    def __call__(
        self,
        t: Float[Tensor, " t"],
    ) -> tuple[
        Float[Tensor, "batch view 4 4"],  # extrinsics
        Float[Tensor, "batch view 3 3"],  # intrinsics
    ]:
        pass


class ModelWrapper(LightningModule):
    logger: Optional[WandbLogger]
    encoder: nn.Module
    encoder_visualizer: Optional[EncoderVisualizer]
    decoder: Decoder
    losses: nn.ModuleList
    optimizer_cfg: OptimizerCfg
    test_cfg: TestCfg
    train_cfg: TrainCfg
    step_tracker: StepTracker | None
    eval_data_cfg: Optional[DatasetCfg | None]

    def __init__(
        self,
        optimizer_cfg: OptimizerCfg,
        test_cfg: TestCfg,
        train_cfg: TrainCfg,
        encoder: Encoder,
        encoder_visualizer: Optional[EncoderVisualizer],
        decoder: Decoder,
        losses: list[Loss],
        step_tracker: StepTracker | None,
        eval_data_cfg: Optional[DatasetCfg | None] = None,
    ) -> None:
        super().__init__()
        self.optimizer_cfg = optimizer_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.step_tracker = step_tracker
        self.eval_data_cfg = eval_data_cfg

        # Set up the model.
        self.encoder = encoder
        self.encoder_visualizer = encoder_visualizer
        self.decoder = decoder
        self.data_shim = get_data_shim(self.encoder)
        self.losses = nn.ModuleList(losses)

        # This is used for testing.
        self.benchmarker = Benchmarker()
        self.eval_cnt = 0

        if self.test_cfg.compute_scores:
            self.test_step_outputs = {}
            self.time_skip_steps_dict = {"encoder": 0, "decoder": 0}

    def training_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        _, _, _, h, w = batch["target"]["image"].shape

        # Run the model.
        gaussians = self.encoder(
            batch["context"], self.global_step, False, scene_names=batch["scene"]
        )

        if isinstance(gaussians, dict):
            pred_depths = gaussians["depths"]
            gaussians = gaussians["gaussians"]

        supervise_intermediate_depth = False

        if gaussians.means.size(0) != batch["target"]["extrinsics"].size(0):
            supervise_intermediate_depth = True
            assert gaussians.means.size(0) % batch["target"]["extrinsics"].size(0) == 0
            num_depths = gaussians.means.size(0) // batch["target"]["extrinsics"].size(
                0
            )
            # add loss to intermediate depth predictions
            target_extrinsics = torch.cat(
                [batch["target"]["extrinsics"]] * num_depths, dim=0
            )
            target_intrinsics = torch.cat(
                [batch["target"]["intrinsics"]] * num_depths, dim=0
            )
            target_near = torch.cat([batch["target"]["near"]] * num_depths, dim=0)
            target_far = torch.cat([batch["target"]["far"]] * num_depths, dim=0)

            output_all = self.decoder.forward(
                gaussians,
                target_extrinsics,
                target_intrinsics,
                target_near,
                target_far,
                (h, w),
                depth_mode=self.train_cfg.depth_mode,
            )
            # split
            batch_size = batch["target"]["extrinsics"].size(0)
            # order: intermediate depth, final depth
            output_intermediate = DecoderOutput(
                color=output_all.color[:-batch_size],
                depth=(
                    output_all.depth[:-batch_size]
                    if output_all.depth is not None
                    else None
                ),
            )
            output = DecoderOutput(
                color=output_all.color[-batch_size:],
                depth=(
                    output_all.depth[-batch_size:]
                    if output_all.depth is not None
                    else None
                ),
            )

        else:
            output = self.decoder.forward(
                gaussians,
                batch["target"]["extrinsics"],
                batch["target"]["intrinsics"],
                batch["target"]["near"],
                batch["target"]["far"],
                (h, w),
                depth_mode=self.train_cfg.depth_mode,
            )

        target_gt = batch["target"]["image"]

        # Compute metrics.
        psnr_probabilistic = compute_psnr(
            rearrange(target_gt, "b v c h w -> (b v) c h w"),
            rearrange(output.color, "b v c h w -> (b v) c h w"),
        )
        self.log("train/psnr", psnr_probabilistic.mean())

        # Compute and log loss.
        total_loss = 0

        valid_depth_mask = None

        for loss_fn in self.losses:
            if loss_fn.name == "mse":
                loss = loss_fn.forward(
                    output,
                    batch,
                    gaussians,
                    self.global_step,
                    l1_loss=self.train_cfg.l1_loss,
                    clamp_large_error=self.train_cfg.train_ignore_large_loss,
                    valid_depth_mask=valid_depth_mask,
                )
            else:
                loss = loss_fn.forward(
                    output,
                    batch,
                    gaussians,
                    self.global_step,
                    valid_depth_mask=valid_depth_mask,
                )
            self.log(f"loss/{loss_fn.name}", loss)
            total_loss = total_loss + loss

        # color loss on intermediate output
        if supervise_intermediate_depth:
            for loss_fn in self.losses:
                if output_intermediate.color.size(0) != batch_size:
                    assert output_intermediate.color.size(0) % batch_size == 0
                    num_intermediate = output_intermediate.color.size(0) // batch_size
                    intermediate_loss = 0
                    for i in range(num_intermediate):
                        curr_output = DecoderOutput(
                            color=output_intermediate.color[
                                (batch_size * i) : (batch_size * (i + 1))
                            ],
                            depth=(
                                output_intermediate.depth[
                                    (batch_size * i) : (batch_size * (i + 1))
                                ]
                                if output_intermediate.depth is not None
                                else None
                            ),
                        )
                        curr_loss_weight = self.train_cfg.intermediate_loss_weight ** (
                            num_intermediate - i
                        )

                        if loss_fn.name == "mse":
                            loss = loss_fn.forward(
                                curr_output,
                                batch,
                                gaussians,
                                self.global_step,
                                l1_loss=self.train_cfg.l1_loss,
                                clamp_large_error=self.train_cfg.train_ignore_large_loss,
                                valid_depth_mask=valid_depth_mask,
                            )
                        else:
                            loss = loss_fn.forward(
                                curr_output,
                                batch,
                                gaussians,
                                self.global_step,
                                valid_depth_mask=valid_depth_mask,
                            )

                        intermediate_loss = intermediate_loss + curr_loss_weight * loss

                    self.log(f"loss/{loss_fn.name}_intermediate", intermediate_loss)
                    total_loss = total_loss + intermediate_loss
                else:
                    if loss_fn.name == "mse":
                        loss = loss_fn.forward(
                            output_intermediate,
                            batch,
                            gaussians,
                            self.global_step,
                            l1_loss=self.train_cfg.l1_loss,
                            clamp_large_error=self.train_cfg.train_ignore_large_loss,
                            valid_depth_mask=valid_depth_mask,
                        )
                    else:
                        loss = loss_fn.forward(
                            output_intermediate,
                            batch,
                            gaussians,
                            self.global_step,
                            valid_depth_mask=valid_depth_mask,
                        )
                    self.log(f"loss/{loss_fn.name}_intermediate", loss)
                    total_loss = (
                        total_loss + self.train_cfg.intermediate_loss_weight * loss
                    )

        self.log("loss/total", total_loss)

        if (
            self.global_rank == 0
            and self.global_step % self.train_cfg.print_log_every_n_steps == 0
        ):
            print(
                f"train step {self.global_step}; "
                f"scene = {[x[:20] for x in batch['scene']]}; "
                f"context = {batch['context']['index'].tolist()}; "
                f"bound = [{batch['context']['near'].detach().cpu().numpy().mean()} "
                f"{batch['context']['far'].detach().cpu().numpy().mean()}]; "
                f"loss = {total_loss:.6f}"
            )
        self.log("info/near", batch["context"]["near"].detach().cpu().numpy().mean())
        self.log("info/far", batch["context"]["far"].detach().cpu().numpy().mean())
        self.log("info/global_step", self.global_step)  # hack for ckpt monitor

        # Tell the data loader processes about the current step.
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)

        if self.global_step == 5 and self.global_rank == 0:
            os.system("nvidia-smi")

        # ignore bad samples
        if self.train_cfg.train_ignore_large_loss > 0:
            if total_loss > self.train_cfg.train_ignore_large_loss:
                return 0.00000001 * total_loss

        return total_loss

    def test_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        b, v, _, h, w = batch["target"]["image"].shape
        assert b == 1

        pred_depths = None

        # save input views for visualization
        if self.test_cfg.save_input_images:
            (scene,) = batch["scene"]
            self.test_cfg.output_path = os.path.join(get_cfg()["output_dir"], "test")
            path = Path(get_cfg()["output_dir"])

            input_images = batch["context"]["image"][0]  # [V, 3, H, W]
            index = batch["context"]["index"][0]
            for idx, color in zip(index, input_images):
                save_image(color, path / scene / f"color/{idx:0>6}.png")

            return

        # save depth vis
        if self.test_cfg.save_depth:
            visualization_dump = {}
        else:
            visualization_dump = None

        # Render Gaussians.
        with self.benchmarker.time("encoder"):
            gaussians = self.encoder(
                batch["context"],
                self.global_step,
                deterministic=False,
                visualization_dump=visualization_dump,
            )

            if isinstance(gaussians, dict):
                pred_depths = gaussians["depths"]
                if "depth" in batch["context"]:
                    depth_gt = batch["context"]["depth"]
                gaussians = gaussians["gaussians"]

        if not self.train_cfg.forward_depth_only:
            with self.benchmarker.time("decoder", num_calls=v):

                camera_poses = batch["target"]["extrinsics"]

                if self.test_cfg.stablize_camera:
                    stable_poses = render_stabilization_path(
                        camera_poses[0].detach().cpu().numpy(),
                        k_size=self.test_cfg.stab_camera_kernel,
                    )

                    stable_poses = list(
                        map(
                            lambda x: np.concatenate(
                                (x, np.array([[0.0, 0.0, 0.0, 1.0]])), axis=0
                            ),
                            stable_poses,
                        )
                    )
                    stable_poses = torch.from_numpy(np.stack(stable_poses, axis=0)).to(
                        camera_poses
                    )
                    camera_poses = stable_poses.unsqueeze(0)

                if self.test_cfg.render_chunk_size is not None:
                    chunk_size = self.test_cfg.render_chunk_size
                    num_chunks = math.ceil(camera_poses.shape[1] / chunk_size)

                    output = None
                    for i in range(num_chunks):
                        start = chunk_size * i
                        end = chunk_size * (i + 1)

                        render_intrinsics = batch["target"]["intrinsics"]
                        render_near = batch["target"]["near"]
                        render_far = batch["target"]["far"]

                        curr_output = self.decoder.forward(
                            gaussians,
                            camera_poses[:, start:end],
                            render_intrinsics[:, start:end],
                            render_near[:, start:end],
                            render_far[:, start:end],
                            (h, w),
                            depth_mode=None,
                        )

                        if i == 0:
                            output = curr_output
                        else:
                            # ignore depth
                            output.color = torch.cat(
                                (output.color, curr_output.color), dim=1
                            )

                else:
                    output = self.decoder.forward(
                        gaussians,
                        camera_poses,
                        batch["target"]["intrinsics"],
                        batch["target"]["near"],
                        batch["target"]["far"],
                        (h, w),
                        depth_mode=None,
                    )

        (scene,) = batch["scene"]
        self.test_cfg.output_path = os.path.join(get_cfg()["output_dir"], "test")
        path = Path(get_cfg()["output_dir"])

        # save depth
        if self.test_cfg.save_depth:
            if self.train_cfg.forward_depth_only:
                depth = pred_depths[0].cpu().detach()  # [V, H, W]
            else:
                depth = (
                    visualization_dump["depth"][0, :, :, :, 0, 0].cpu().detach()
                )  # [V, H, W]

            index = batch["context"]["index"][0]

            if self.test_cfg.save_depth_concat_img:
                # concat (img0, img1, depth0, depth1)
                image = batch['context']['image'][0]  # [V, 3, H, W] in [0,1]
                image = rearrange(image, "b c h w -> h (b w) c")  # [H, VW, 3]
                image_concat = (image.detach().cpu().numpy() * 255).astype(np.uint8)  # [H, VW, 3]

                depth_concat = []

            for idx, depth_i in zip(index, depth):
                depth_viz = viz_depth_tensor(
                    1.0 / depth_i, return_numpy=True
                )  # [H, W, 3]

                if self.test_cfg.save_depth_concat_img:
                    depth_concat.append(depth_viz)
                else:
                    save_path = path / scene / f"{idx:0>6}.png"
                    save_dir = os.path.dirname(save_path)
                    os.makedirs(save_dir, exist_ok=True)
                    Image.fromarray(depth_viz).save(save_path)

                # save depth as npy
                if self.test_cfg.save_depth_npy:
                    depth_npy = depth_i.deach().cpu().numpy()
                    save_path = path / scene / f"{idx:0>6}.npy"
                    np.save(save_path, depth_npy)

            if self.test_cfg.save_depth_concat_img:
                depth_concat = np.concatenate(depth_concat, axis=1)  # [H, VW, 3]
                concat = np.concatenate((image_concat, depth_concat), axis=0)  # [2H, VW, 3]

                save_path = path / f"{scene}.png"
                path.mkdir(parents=True, exist_ok=True)
                Image.fromarray(concat).save(save_path)

            if self.train_cfg.forward_depth_only:
                return

        images_prob = output.color[0]
        rgb_gt = batch["target"]["image"][0]

        # Save images.
        if self.test_cfg.save_image:
            if self.test_cfg.save_gt_image:
                for index, color, gt in zip(
                    batch["target"]["index"][0], images_prob, rgb_gt
                ):
                    save_image(color, path / scene / f"color/{index:0>6}.png")
                    save_image(gt, path / scene / f"color/{index:0>6}_gt.png")
            else:
                for index, color in zip(batch["target"]["index"][0], images_prob):
                    save_image(color, path / scene / f"color/{index:0>6}.png")

        # save video
        if self.test_cfg.save_video:
            frame_str = "_".join([str(x.item()) for x in batch["context"]["index"][0]])
            save_video(
                [a for a in images_prob],
                path / "video" / f"{scene}_frame_{frame_str}.mp4",
            )

        # compute scores
        if self.test_cfg.compute_scores:
            if batch_idx < self.test_cfg.eval_time_skip_steps:
                self.time_skip_steps_dict["encoder"] += 1
                self.time_skip_steps_dict["decoder"] += v

            if not self.train_cfg.forward_depth_only:
                rgb = images_prob

                if f"psnr" not in self.test_step_outputs:
                    self.test_step_outputs[f"psnr"] = []
                if f"ssim" not in self.test_step_outputs:
                    self.test_step_outputs[f"ssim"] = []
                if f"lpips" not in self.test_step_outputs:
                    self.test_step_outputs[f"lpips"] = []

                self.test_step_outputs[f"psnr"].append(
                    compute_psnr(rgb_gt, rgb).mean().item()
                )
                self.test_step_outputs[f"ssim"].append(
                    compute_ssim(rgb_gt, rgb).mean().item()
                )
                self.test_step_outputs[f"lpips"].append(
                    compute_lpips(rgb_gt, rgb).mean().item()
                )

    def on_test_end(self) -> None:
        out_dir = Path(self.test_cfg.output_path)
        saved_scores = {}
        if self.test_cfg.compute_scores:
            self.benchmarker.dump_memory(out_dir / "peak_memory.json")
            self.benchmarker.dump(out_dir / "benchmark.json")

            for metric_name, metric_scores in self.test_step_outputs.items():
                avg_scores = sum(metric_scores) / len(metric_scores)
                saved_scores[metric_name] = avg_scores
                print(metric_name, avg_scores)
                with (out_dir / f"scores_{metric_name}_all.json").open("w") as f:
                    json.dump(metric_scores, f)
                metric_scores.clear()

            for tag, times in self.benchmarker.execution_times.items():
                times = times[int(self.time_skip_steps_dict[tag]) :]
                saved_scores[tag] = [len(times), np.mean(times)]
                print(
                    f"{tag}: {len(times)} calls, avg. {np.mean(times)} seconds per call"
                )
                self.time_skip_steps_dict[tag] = 0

            with (out_dir / f"scores_all_avg.json").open("w") as f:
                json.dump(saved_scores, f)
            self.benchmarker.clear_history()
        else:
            self.benchmarker.dump(out_dir / "benchmark.json")
            self.benchmarker.dump_memory(out_dir / "peak_memory.json")
            self.benchmarker.summarize()

    @rank_zero_only
    def validation_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)

        if self.global_rank == 0:
            print(
                f"validation step {self.global_step}; "
                f"scene = {[a[:20] for a in batch['scene']]}; "
                f"context = {batch['context']['index'].tolist()}"
            )

        # Render Gaussians.
        b, _, _, h, w = batch["target"]["image"].shape
        assert b == 1
        gaussians_softmax = self.encoder(
            batch["context"],
            self.global_step,
            deterministic=False,
        )

        pred_depths = None

        if isinstance(gaussians_softmax, dict):
            pred_depths = gaussians_softmax["depths"]
            if "depth" in batch["context"]:
                depth_gt = batch["context"]["depth"]  # [B, V, H, W]
            gaussians_softmax = gaussians_softmax["gaussians"]

        if not self.train_cfg.forward_depth_only:
            output_softmax = self.decoder.forward(
                gaussians_softmax,
                batch["target"]["extrinsics"],
                batch["target"]["intrinsics"],
                batch["target"]["near"],
                batch["target"]["far"],
                (h, w),
            )
            rgb_softmax = output_softmax.color[0]

            # Compute validation metrics.
            rgb_gt = batch["target"]["image"][0]
            for tag, rgb in zip(("val",), (rgb_softmax,)):
                psnr = compute_psnr(rgb_gt, rgb).mean()
                self.log(f"val/psnr_{tag}", psnr)
                lpips = compute_lpips(rgb_gt, rgb).mean()
                self.log(f"val/lpips_{tag}", lpips)
                ssim = compute_ssim(rgb_gt, rgb).mean()
                self.log(f"val/ssim_{tag}", ssim)

        # viz depth
        if pred_depths is not None:
            # only visualize predicted depth
            pred_depths = pred_depths[0]  # [V, H, W]

            # gaussian downsample
            if pred_depths.shape[1:] != batch["context"]["image"].shape[-2:]:
                pred_depths = F.interpolate(
                    pred_depths.unsqueeze(1),
                    size=batch["context"]["image"].shape[-2:],
                    mode="bilinear",
                    align_corners=True,
                ).squeeze(1)

            inverse_depth_pred = 1.0 / pred_depths

            concat = []
            for i in range(inverse_depth_pred.size(0)):
                concat.append(inverse_depth_pred[i])

            concat = torch.cat(concat, dim=1)  # [H, W*N]

            depth_viz = viz_depth_tensor(concat.cpu().detach())  # [3, H, W*N]

            # also concat images
            input_images = batch["context"]["image"][0]  # [N, 3, H, W]
            concat_img = [img for img in input_images]
            concat_img = torch.cat(concat_img, dim=-1) * 255  # [3, H, W*N]

            concat = torch.cat(
                (concat_img.cpu().detach(), depth_viz), dim=1
            )  # [3, H*2, W*N]

            self.logger.log_image(
                "depth",
                [concat],
                step=self.global_step,
                caption=batch["scene"],
            )

        if not self.train_cfg.forward_depth_only:
            # Construct comparison image.
            comparison = hcat(
                add_label(vcat(*batch["context"]["image"][0]), "Context"),
                add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
                add_label(vcat(*rgb_softmax), "Target (Prediction)"),
            )
            self.logger.log_image(
                "comparison",
                [prep_image(add_border(comparison))],
                step=self.global_step,
                caption=batch["scene"],
            )

            if not self.train_cfg.no_log_projections:
                # Render projections and construct projection image.
                projections = hcat(
                    *render_projections(
                        gaussians_softmax,
                        256,
                        extra_label="(Prediction)",
                    )[0]
                )
                self.logger.log_image(
                    "projection",
                    [prep_image(add_border(projections))],
                    step=self.global_step,
                )

                # Draw cameras.
                cameras = hcat(*render_cameras(batch, 256))
                self.logger.log_image(
                    "cameras", [prep_image(add_border(cameras))], step=self.global_step
                )

            if self.encoder_visualizer is not None:
                for k, image in self.encoder_visualizer.visualize(
                    batch["context"], self.global_step
                ).items():
                    self.logger.log_image(k, [prep_image(image)], step=self.global_step)

            # Run video validation step.
            if not self.train_cfg.no_viz_video:
                self.render_video_interpolation(batch)
                self.render_video_wobble(batch)
                if self.train_cfg.extended_visualization:
                    self.render_video_interpolation_exaggerated(batch)

    def on_validation_epoch_end(self) -> None:
        """hack to run the full validation"""
        if self.trainer.sanity_checking and self.global_rank == 0:
            print(self.encoder)  # log the model to wandb log files

        if (not self.trainer.sanity_checking) and (self.eval_data_cfg is not None):
            self.eval_cnt = self.eval_cnt + 1
            if self.eval_cnt % self.train_cfg.eval_model_every_n_val == 0:
                # backup current ckpt before running full test sets eval
                if self.train_cfg.eval_save_model:
                    ckpt_saved_path = (
                        self.trainer.checkpoint_callback.format_checkpoint_name(
                            dict(
                                epoch=self.trainer.current_epoch,
                                step=self.trainer.global_step,
                            )
                        )
                    )
                    backup_dir = str(
                        Path(ckpt_saved_path).parent.parent / "checkpoints_backups"
                    )
                    if self.global_rank == 0:
                        os.makedirs(backup_dir, exist_ok=True)
                    ckpt_saved_path = os.path.join(
                        backup_dir, os.path.basename(ckpt_saved_path)
                    )
                    # call save_checkpoint on ALL process as suggested by pytorch_lightning
                    self.trainer.save_checkpoint(
                        ckpt_saved_path,
                        weights_only=True,
                    )
                    if self.global_rank == 0:
                        print(f"backup model to {ckpt_saved_path}.")

                # run full test sets eval on rank=0 device
                self.run_full_test_sets_eval()

    @rank_zero_only
    def run_full_test_sets_eval(self) -> None:
        start_t = time.time()

        pred_depths = None
        depth_gt = None

        full_testsets = self.trainer.datamodule.test_dataloader(
            dataset_cfg=self.eval_data_cfg
        )
        scores_dict = {}

        if not self.train_cfg.forward_depth_only:
            for score_tag in ("psnr", "ssim", "lpips"):
                scores_dict[score_tag] = {}
                for method_tag in ("deterministic", "probabilistic"):
                    scores_dict[score_tag][method_tag] = []

        # evaluate depth
        if self.train_cfg.viz_depth:
            for score_tag in ("abs_rel", "rmse", "a1"):
                scores_dict[score_tag] = {}
                for method_tag in ("deterministic", "probabilistic"):
                    scores_dict[score_tag][method_tag] = []

        self.benchmarker.clear_history()
        time_skip_first_n_steps = min(
            self.train_cfg.eval_time_skip_steps, len(full_testsets)
        )
        time_skip_steps_dict = {"encoder": 0, "decoder": 0}
        for batch_idx, batch in tqdm(
            enumerate(full_testsets),
            total=min(len(full_testsets), self.train_cfg.eval_data_length),
        ):
            if batch_idx >= self.train_cfg.eval_data_length:
                break

            batch = self.data_shim(batch)
            batch = self.transfer_batch_to_device(batch, "cuda", dataloader_idx=0)

            # Render Gaussians.
            b, v, _, h, w = batch["target"]["image"].shape
            assert b == 1
            if batch_idx < time_skip_first_n_steps:
                time_skip_steps_dict["encoder"] += 1
                time_skip_steps_dict["decoder"] += v

            with self.benchmarker.time("encoder"):
                gaussians_probabilistic = self.encoder(
                    batch["context"],
                    self.global_step,
                    deterministic=False,
                )

                if isinstance(gaussians_probabilistic, dict):
                    pred_depths = gaussians_probabilistic["depths"]
                    if "depth" in batch["context"]:
                        depth_gt = batch["context"]["depth"]
                    gaussians_probabilistic = gaussians_probabilistic["gaussians"]

            if not self.train_cfg.forward_depth_only:
                with self.benchmarker.time("decoder", num_calls=v):
                    output_probabilistic = self.decoder.forward(
                        gaussians_probabilistic,
                        batch["target"]["extrinsics"],
                        batch["target"]["intrinsics"],
                        batch["target"]["near"],
                        batch["target"]["far"],
                        (h, w),
                    )
                rgbs = [output_probabilistic.color[0]]
                tags = ["probabilistic"]

                if self.train_cfg.eval_deterministic:
                    gaussians_deterministic = self.encoder(
                        batch["context"],
                        self.global_step,
                        deterministic=True,
                    )
                    output_deterministic = self.decoder.forward(
                        gaussians_deterministic,
                        batch["target"]["extrinsics"],
                        batch["target"]["intrinsics"],
                        batch["target"]["near"],
                        batch["target"]["far"],
                        (h, w),
                    )
                    rgbs.append(output_deterministic.color[0])
                    tags.append("deterministic")

                # Compute validation metrics.
                rgb_gt = batch["target"]["image"][0]
                for tag, rgb in zip(tags, rgbs):
                    scores_dict["psnr"][tag].append(
                        compute_psnr(rgb_gt, rgb).mean().item()
                    )
                    scores_dict["lpips"][tag].append(
                        compute_lpips(rgb_gt, rgb).mean().item()
                    )
                    scores_dict["ssim"][tag].append(
                        compute_ssim(rgb_gt, rgb).mean().item()
                    )

        # summarise scores and log to logger
        for score_tag, methods in scores_dict.items():
            for method_tag, cur_scores in methods.items():
                if len(cur_scores) > 0:
                    cur_mean = sum(cur_scores) / len(cur_scores)
                    self.log(f"test/{score_tag}", cur_mean)
        # summarise run time
        for tag, times in self.benchmarker.execution_times.items():
            times = times[int(time_skip_steps_dict[tag]) :]
            print(f"{tag}: {len(times)} calls, avg. {np.mean(times)} seconds per call")
            self.log(f"test/runtime_avg_{tag}", np.mean(times))
        self.benchmarker.clear_history()

        overall_eval_time = time.time() - start_t
        print(f"Eval total time cost: {overall_eval_time:.3f}s")
        self.log("test/runtime_all", overall_eval_time)

    @rank_zero_only
    def render_video_wobble(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            extrinsics = generate_wobble(
                batch["context"]["extrinsics"][:, 0],
                delta * 0.25,
                t,
            )
            intrinsics = repeat(
                batch["context"]["intrinsics"][:, 0],
                "b i j -> b v i j",
                v=t.shape[0],
            )
            return extrinsics, intrinsics

        return self.render_video_generic(batch, trajectory_fn, "wobble", num_frames=60)

    @rank_zero_only
    def render_video_interpolation(self, batch: BatchedExample) -> None:
        _, v, _, _ = batch["context"]["extrinsics"].shape

        def trajectory_fn(t):
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["extrinsics"][0, 0]
                ),
                t,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["intrinsics"][0, 0]
                ),
                t,
            )
            return extrinsics[None], intrinsics[None]

        return self.render_video_generic(batch, trajectory_fn, "rgb")

    @rank_zero_only
    def render_video_interpolation_exaggerated(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            tf = generate_wobble_transformation(
                delta * 0.5,
                t,
                5,
                scale_radius_with_t=False,
            )
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["extrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["intrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            return extrinsics @ tf, intrinsics[None]

        return self.render_video_generic(
            batch,
            trajectory_fn,
            "interpolation_exagerrated",
            num_frames=300,
            smooth=False,
            loop_reverse=False,
        )

    @rank_zero_only
    def render_video_generic(
        self,
        batch: BatchedExample,
        trajectory_fn: TrajectoryFn,
        name: str,
        num_frames: int = 30,
        smooth: bool = True,
        loop_reverse: bool = True,
    ) -> None:
        # Render probabilistic estimate of scene.
        gaussians_prob = self.encoder(batch["context"], self.global_step, False)
        # gaussians_det = self.encoder(batch["context"], self.global_step, True)

        if isinstance(gaussians_prob, dict):
            gaussians_prob = gaussians_prob["gaussians"]

        t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=self.device)
        if smooth:
            t = (torch.cos(torch.pi * (t + 1)) + 1) / 2

        extrinsics, intrinsics = trajectory_fn(t)

        _, _, _, h, w = batch["context"]["image"].shape

        # Color-map the result.
        def depth_map(result):
            near = result[result > 0][:16_000_000].quantile(0.01).log()
            far = result.view(-1)[:16_000_000].quantile(0.99).log()
            result = result.log()
            result = 1 - (result - near) / (far - near)
            return apply_color_map_to_image(result, "turbo")

        near = repeat(batch["context"]["near"][:, 0], "b -> b v", v=num_frames)
        far = repeat(batch["context"]["far"][:, 0], "b -> b v", v=num_frames)
        output_prob = self.decoder.forward(
            gaussians_prob, extrinsics, intrinsics, near, far, (h, w), "depth"
        )
        images_prob = [
            vcat(rgb, depth)
            for rgb, depth in zip(output_prob.color[0], depth_map(output_prob.depth[0]))
        ]

        images = [
            add_border(
                hcat(
                    add_label(image_prob, "Prediction"),
                )
            )
            for image_prob, _ in zip(images_prob, images_prob)
        ]

        video = torch.stack(images)
        video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        if loop_reverse:
            video = pack([video, video[::-1][1:-1]], "* c h w")[0]
        visualizations = {
            f"video/{name}": wandb.Video(video[None], fps=30, format="mp4")
        }

        # Since the PyTorch Lightning doesn't support video logging, log to wandb directly.
        try:
            wandb.log(visualizations)
        except Exception:
            assert isinstance(self.logger, LocalLogger)
            for key, value in visualizations.items():
                tensor = value._prepare_video(value.data)
                clip = mpy.ImageSequenceClip(list(tensor), fps=value._fps)
                dir = LOG_PATH / key
                dir.mkdir(exist_ok=True, parents=True)
                clip.write_videofile(
                    str(dir / f"{self.global_step:0>6}.mp4"), logger=None
                )

    def configure_optimizers(self):
        if self.optimizer_cfg.lr_monodepth > 0:
            pretrained_params = []
            new_params = []

            for name, param in self.named_parameters():
                if "pretrained" in name:
                    pretrained_params.append(param)
                else:
                    new_params.append(param)

            optimizer = torch.optim.AdamW(
                [
                    {
                        "params": pretrained_params,
                        "lr": self.optimizer_cfg.lr_monodepth,
                    },
                    {"params": new_params, "lr": self.optimizer_cfg.lr},
                ],
                weight_decay=self.optimizer_cfg.weight_decay,
            )

            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                [self.optimizer_cfg.lr_monodepth, self.optimizer_cfg.lr],
                self.trainer.max_steps + 10,
                pct_start=0.01,
                cycle_momentum=False,
                anneal_strategy="cos",
            )

        else:
            optimizer = optim.AdamW(
                self.parameters(),
                lr=self.optimizer_cfg.lr,
                weight_decay=self.optimizer_cfg.weight_decay,
            )

            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                self.optimizer_cfg.lr,
                self.trainer.max_steps + 10,
                pct_start=0.01,
                cycle_momentum=False,
                anneal_strategy="cos",
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
