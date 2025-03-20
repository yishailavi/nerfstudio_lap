from dataclasses import dataclass, field
from typing import Type, Dict, List, Optional, Union, Tuple

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.model_components.lib_bilagrid import color_correct, total_variation_loss
from torch import Tensor
from torch.nn import Parameter
import torch
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig, resize_image, get_viewmat

from splatfacto_lap.rasterization import rasterization
from splatfacto_lap.splatfacto_lap_strategy import SplatfactoLapStrategy
from splatfacto_lap.utils import gauss_kernel, downsample_antialias_n_times, downsample_n_times_upsample_n_times


@dataclass
class SplatfactoModelLapConfig(SplatfactoModelConfig):
    _target: Type = field(default_factory=lambda: SplatfactoModelLap)

    num_laplacian_levels: int = 3
    add_level_every: int = 3000
    disable_refinement_length: int = 300
    gauss_loss_lambda: float = 1.
    lower_gauss_loss_lambda_factor: float = 0.1
    gauss_loss_fft_lambda: float = 0.001


class SplatfactoModelLap(SplatfactoModel):

    config: SplatfactoModelLapConfig

    def populate_modules(self):
        super().populate_modules()
        num_points = self.gauss_params["means"].shape[0]
        self.gauss_params["levels"] = torch.zeros(num_points)

        self.info = dict()
        self.num_levels = 1
        self.output_gaussian_levels = True
        self.output_laplacian_levels = True
        self.update_optimizers = False
        self.disable_refinement_counter = 0

        self.strategy = SplatfactoLapStrategy(
            prune_opa=self.config.cull_alpha_thresh,
            grow_grad2d=self.config.densify_grad_thresh,
            grow_scale3d=self.config.densify_size_thresh,
            grow_scale2d=self.config.split_screen_size,
            prune_scale3d=self.config.cull_scale_thresh,
            prune_scale2d=self.config.cull_screen_size,
            refine_scale2d_stop_iter=self.config.stop_screen_size_at,
            refine_start_iter=self.config.warmup_length,
            refine_stop_iter=self.config.stop_split_at,
            reset_every=self.config.reset_alpha_every * self.config.refine_every,
            refine_every=self.config.refine_every,
            pause_refine_after_reset=self.num_train_data + self.config.refine_every,
            absgrad=self.config.use_absgrad,
            revised_opacity=False,
            verbose=True,
        )
        self.strategy_state = self.strategy.initialize_state(scene_scale=1.0)


    @property
    def levels(self):
        return self.gauss_params["levels"]

    def load_state_dict(self, dict, **kwargs):  # type: ignore
        # resize the parameters to match the new number of points
        self.step = 30000
        self.num_levels = self.config.num_laplacian_levels
        if "means" in dict:
            # For backwards compatibility, we remap the names of parameters from
            # means->gauss_params.means since old checkpoints have that format
            for p in ["means", "scales", "quats", "features_dc", "features_rest", "opacities", "levels"]:
                dict[f"gauss_params.{p}"] = dict[p]
        newp = dict["gauss_params.means"].shape[0]
        for name, param in self.gauss_params.items():
            old_shape = param.shape
            new_shape = (newp,) + old_shape[1:]
            self.gauss_params[name] = torch.nn.Parameter(torch.zeros(new_shape, device=self.device))
        super().load_state_dict(dict, **kwargs)

    def step_post_backward(self, step):
        if self.update_optimizers:
            return
        assert step == self.step
        if self.disable_refinement_counter > 0:
            self.strategy.disable_refinement()
            self.disable_refinement_counter -= 1
        else:
            self.strategy.enable_refinement()
        self.strategy.step_post_backward(
            params=self.gauss_params,
            optimizers=self.optimizers,
            state=self.strategy_state,
            step=self.step,
            info=self.info,
            packed=False,
        )

    def step_cb(self, optimizers: Optimizers, step):
        self.step = step
        if self.update_optimizers:
            self.update_optimizers = False
            new_optimizers = Optimizers(optimizers.config, self.get_param_groups())
            optimizers.optimizers = new_optimizers.optimizers
            optimizers.schedulers = new_optimizers.schedulers
            optimizers.parameters = new_optimizers.parameters
        self.optimizers = optimizers.optimizers
        return optimizers

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        # Here we explicitly use the means, scales as parameters so that the user can override this function and
        # specify more if they want to add more optimizable params to gaussians.
        return {
            name: [self.gauss_params[name]]
            for name in ["means", "scales", "quats", "features_dc", "features_rest", "opacities", "levels"]
        }

    def _get_downscale_factor(self):
        return 1

    def _downscale_if_required(self, image):
        d = self._get_downscale_factor() * (2 ** (self.config.num_laplacian_levels - self.num_levels))
        if d > 1:
            return resize_image(image, d)
        return image

    def _update_levels(self):
        if self.num_levels >= self.config.num_laplacian_levels:
            return
        if self.num_levels >= (self.step // self.config.add_level_every) + 1:
            return
        else:
            self.num_levels += 1

            means = self.means.clone()
            num_points = means.shape[0]
            scales = self.scales.clone()
            quats = self.quats.clone()
            features_dc = self.features_dc.clone()
            features_rest = self.features_rest.clone()
            opacities = self.opacities.clone()
            update_params = {
                "means": means.to('cuda'),
                "scales": scales.to('cuda'),
                "quats": quats.to('cuda'),
                "features_dc": features_dc.to('cuda'),
                "features_rest": features_rest.to('cuda'),
                "opacities": opacities.to('cuda'),
                "levels": (torch.ones(num_points) * (self.num_levels - 1)).to('cuda')
            }

            def param_fn(name: str, p):
                return torch.nn.Parameter(torch.cat([p, update_params[name]], dim=0))

            def optimizer_fn(key: str, v):
                return torch.cat([v, torch.zeros((len(means), *v.shape[1:]), device=self.device)])

            for name in update_params.keys():
                optimizer = self.optimizers[name]
                for i, param_group in enumerate(optimizer.param_groups):
                    p = param_group["params"][0]
                    p_state = optimizer.state[p]
                    del optimizer.state[p]
                    for key in p_state.keys():
                        if key != "step":
                            v = p_state[key]
                            p_state[key] = optimizer_fn(key, v)
                    p_new = param_fn(name, p)
                    optimizer.param_groups[i]["params"] = [p_new]
                    optimizer.state[p_new] = p_state
                    self.gauss_params[name] = p_new

            self.update_optimizers = True
            self.disable_refinement_counter = self.config.disable_refinement_length

    def get_outputs(self, camera: Cameras, bin_mask: Optional[Tensor] = None,
                    boolean_gauss_mask: Optional[Tensor] = None, num_levels_to_render=None,
                    fov_mask=None) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a camera and returns a dictionary of outputs.

        Args:
            camera: The camera(s) for which output images are rendered. It should have
            all the needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        if num_levels_to_render is None:
            num_levels_to_render = self.num_levels

        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}

        if self.training:
            assert camera.shape[0] == 1, "Only one camera at a time"
            optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera)
        else:
            optimized_camera_to_world = camera.camera_to_worlds

        self._update_levels()

        # cropping
        assert not self.crop_box, "crop box is not supported at the moment"
        # if self.crop_box is not None and not self.training:
        #     crop_ids = self.crop_box.within(self.means).squeeze()
        #     if crop_ids.sum() == 0:
        #         return self.get_empty_outputs(
        #             int(camera.width.item()), int(camera.height.item()), self.background_color
        #         )
        # else:
        #     crop_ids = None

        crop_ids = boolean_gauss_mask
        if crop_ids is not None:
            crop_ids &= self.levels < num_levels_to_render
        else:
            crop_ids = self.levels < num_levels_to_render

        if crop_ids is not None:
            opacities_crop = self.opacities[crop_ids]
            means_crop = self.means[crop_ids]
            features_dc_crop = self.features_dc[crop_ids]
            features_rest_crop = self.features_rest[crop_ids]
            scales_crop = self.scales[crop_ids]
            quats_crop = self.quats[crop_ids]
            levels_crop = self.levels[crop_ids]
        else:
            opacities_crop = self.opacities
            means_crop = self.means
            features_dc_crop = self.features_dc
            features_rest_crop = self.features_rest
            scales_crop = self.scales
            quats_crop = self.quats
            levels_crop = self.levels

        colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)

        camera_scale_fac = self._get_downscale_factor() * (2 ** (self.config.num_laplacian_levels - self.num_levels))
        camera.rescale_output_resolution(1 / camera_scale_fac)
        viewmat = get_viewmat(optimized_camera_to_world)
        K = camera.get_intrinsics_matrices().cuda()
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)
        camera.rescale_output_resolution(camera_scale_fac)  # type: ignore

        # apply the compensation of screen space blurring to gaussians
        if self.config.rasterize_mode not in ["antialiased", "classic"]:
            raise ValueError("Unknown rasterize_mode: %s", self.config.rasterize_mode)

        if self.config.output_depth_during_training or not self.training:
            render_mode = "RGB+ED"
        else:
            render_mode = "RGB"

        if self.config.sh_degree > 0:
            sh_degree_to_use = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
        else:
            colors_crop = torch.sigmoid(colors_crop).squeeze(1)  # [N, 1, 3] -> [N, 3]
            sh_degree_to_use = None

        render, alpha, self.info = rasterization(
            means=means_crop,
            quats=quats_crop,  # rasterization does normalization internally
            scales=torch.exp(scales_crop),
            opacities=torch.sigmoid(opacities_crop).squeeze(-1),
            colors=colors_crop,
            viewmats=viewmat,  # [1, 4, 4]
            Ks=K,  # [1, 3, 3]
            width=W,
            height=H,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode=render_mode,
            sh_degree=sh_degree_to_use,
            sparse_grad=False,
            absgrad=self.strategy.absgrad,
            rasterize_mode=self.config.rasterize_mode,
            levels=levels_crop,
            bin_mask=bin_mask,
            fov_mask=fov_mask
        )
        if self.training:
            self.strategy.step_pre_backward(
                self.gauss_params, self.optimizers, self.strategy_state, self.step, self.info
            )
        alpha = alpha[:, ...]

        background = self._get_background_color()
        rgb = render[:, ..., :3] + (1 - alpha) * background
        rgb = torch.clamp(rgb, 0.0, 1.0)

        # apply bilateral grid
        if self.config.use_bilateral_grid and self.training:
            if camera.metadata is not None and "cam_idx" in camera.metadata:
                rgb = self._apply_bilateral_grid(rgb, camera.metadata["cam_idx"], H, W)

        if render_mode == "RGB+ED":
            depth_im = render[:, ..., 3:4]
            depth_im = torch.where(alpha > 0, depth_im, depth_im.detach().max()).squeeze(0)
        else:
            depth_im = None

        if background.shape[0] == 3 and not self.training:
            background = background.expand(H, W, 3)

        ret_dict = {
            f"gaussian_pyramid_{str(self.num_levels - 1)}": rgb.squeeze(0),  # type: ignore,
            "rgb": rgb.squeeze(0),
            "depth": depth_im,  # type: ignore
            "accumulation": alpha.squeeze(0),  # type: ignore
            "background": background,  # type: ignore
        }  # type: ignore

        if self.output_gaussian_levels:
            for level in range(self.num_levels - 1):
                if self.crop_box is not None and not self.training:
                    crop_ids = self.crop_box.within(self.means).squeeze()
                    if crop_ids.sum() == 0:
                        return self.get_empty_outputs(
                            int(camera.width.item()), int(camera.height.item()), self.background_color
                        )
                else:
                    crop_ids = None

                if crop_ids is None:
                    crop_ids = self.levels <= level
                else:
                    crop_ids &= self.levels == level
                if crop_ids is not None:
                    opacities_crop = self.opacities[crop_ids]
                    means_crop = self.means[crop_ids]
                    features_dc_crop = self.features_dc[crop_ids]
                    features_rest_crop = self.features_rest[crop_ids]
                    scales_crop = self.scales[crop_ids]
                    quats_crop = self.quats[crop_ids]
                    levels_crop = self.levels[crop_ids]
                else:
                    opacities_crop = self.opacities
                    means_crop = self.means
                    features_dc_crop = self.features_dc
                    features_rest_crop = self.features_rest
                    scales_crop = self.scales
                    quats_crop = self.quats
                    levels_crop = self.levels

                colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)
                camera_scale_fac = self._get_downscale_factor() * (2 ** (self.config.num_laplacian_levels - self.num_levels))
                camera.rescale_output_resolution(1 / camera_scale_fac)
                viewmat = get_viewmat(optimized_camera_to_world)
                K = camera.get_intrinsics_matrices().cuda()
                W, H = int(camera.width.item()), int(camera.height.item())
                self.last_size = (H, W)
                camera.rescale_output_resolution(camera_scale_fac)  # type: ignore

                # apply the compensation of screen space blurring to gaussians
                if self.config.rasterize_mode not in ["antialiased", "classic"]:
                    raise ValueError("Unknown rasterize_mode: %s", self.config.rasterize_mode)

                if self.config.output_depth_during_training or not self.training:
                    render_mode = "RGB+ED"
                else:
                    render_mode = "RGB"

                if self.config.sh_degree > 0:
                    sh_degree_to_use = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
                else:
                    colors_crop = torch.sigmoid(colors_crop).squeeze(1)  # [N, 1, 3] -> [N, 3]
                    sh_degree_to_use = None

                render, alpha, _ = rasterization(
                    means=means_crop,
                    quats=quats_crop,  # rasterization does normalization internally
                    scales=torch.exp(scales_crop),
                    opacities=torch.sigmoid(opacities_crop).squeeze(-1),
                    colors=colors_crop,
                    viewmats=viewmat,  # [1, 4, 4]
                    Ks=K,  # [1, 3, 3]
                    width=W,
                    height=H,
                    packed=False,
                    near_plane=0.01,
                    far_plane=1e10,
                    render_mode=render_mode,
                    sh_degree=sh_degree_to_use,
                    sparse_grad=False,
                    absgrad=self.strategy.absgrad,
                    rasterize_mode=self.config.rasterize_mode,
                    levels=levels_crop
                )
                alpha = alpha[:, ...]
                background = self._get_background_color()
                rgb_gauss = render[:, ..., :3] + (1 - alpha) * background
                rgb_gauss = torch.clamp(rgb_gauss, 0.0, 1.0)
                ret_dict[f'gaussian_pyramid_{str(level)}'] = rgb_gauss.squeeze(0)

        if self.output_laplacian_levels:
            for level in range(1, self.num_levels):
                with torch.no_grad():
                    if self.crop_box is not None and not self.training:
                        crop_ids = self.crop_box.within(self.means).squeeze()
                        if crop_ids.sum() == 0:
                            return self.get_empty_outputs(
                                int(camera.width.item()), int(camera.height.item()), self.background_color
                            )
                    else:
                        crop_ids = None

                    if crop_ids is None:
                        crop_ids = self.levels == level
                    else:
                        crop_ids &= self.levels == level
                    if crop_ids is not None:
                        opacities_crop = self.opacities[crop_ids]
                        means_crop = self.means[crop_ids]
                        features_dc_crop = self.features_dc[crop_ids]
                        features_rest_crop = self.features_rest[crop_ids]
                        scales_crop = self.scales[crop_ids]
                        quats_crop = self.quats[crop_ids]
                    else:
                        opacities_crop = self.opacities
                        means_crop = self.means
                        features_dc_crop = self.features_dc
                        features_rest_crop = self.features_rest
                        scales_crop = self.scales
                        quats_crop = self.quats

                    colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)
                    camera_scale_fac = self._get_downscale_factor() * (2 ** (self.config.num_laplacian_levels - self.num_levels))
                    camera.rescale_output_resolution(1 / camera_scale_fac)
                    viewmat = get_viewmat(optimized_camera_to_world)
                    K = camera.get_intrinsics_matrices().cuda()
                    W, H = int(camera.width.item()), int(camera.height.item())
                    self.last_size = (H, W)
                    camera.rescale_output_resolution(camera_scale_fac)  # type: ignore

                    # apply the compensation of screen space blurring to gaussians
                    if self.config.rasterize_mode not in ["antialiased", "classic"]:
                        raise ValueError("Unknown rasterize_mode: %s", self.config.rasterize_mode)

                    if self.config.output_depth_during_training or not self.training:
                        render_mode = "RGB+ED"
                    else:
                        render_mode = "RGB"

                    if self.config.sh_degree > 0:
                        sh_degree_to_use = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
                    else:
                        colors_crop = torch.sigmoid(colors_crop).squeeze(1)  # [N, 1, 3] -> [N, 3]
                        sh_degree_to_use = None

                    levels = torch.ones(len(means_crop), device=self.device) * level
                    render, alpha, _ = rasterization(
                        means=means_crop,
                        quats=quats_crop,  # rasterization does normalization internally
                        scales=torch.exp(scales_crop),
                        opacities=torch.sigmoid(opacities_crop).squeeze(-1),
                        colors=colors_crop,
                        viewmats=viewmat,  # [1, 4, 4]
                        Ks=K,  # [1, 3, 3]
                        width=W,
                        height=H,
                        packed=False,
                        near_plane=0.01,
                        far_plane=1e10,
                        render_mode=render_mode,
                        sh_degree=sh_degree_to_use,
                        sparse_grad=False,
                        absgrad=self.strategy.absgrad,
                        rasterize_mode=self.config.rasterize_mode,
                        levels=levels
                    )
                    rgb_lap = (render[:, ..., :3] + 1) / 2
                    rgb_lap = torch.clamp(rgb_lap, 0.0, 1.0)
                    ret_dict[f'laplacian_pyramid_{str(level)}'] = rgb_lap.squeeze(0)

        return ret_dict

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        metrics_dict = {}
        predicted_rgb = outputs["rgb"]
        assert abs(gt_rgb.shape[0] - predicted_rgb.shape[0]) < 10
        assert abs(gt_rgb.shape[1] - predicted_rgb.shape[1]) < 10
        gt_rgb = gt_rgb[:predicted_rgb.shape[0], :predicted_rgb.shape[1], :]

        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)
        if self.config.color_corrected_metrics:
            cc_rgb = color_correct(predicted_rgb, gt_rgb)
            metrics_dict["cc_psnr"] = self.psnr(cc_rgb, gt_rgb)

        metrics_dict["gaussian_count"] = self.num_points

        self.camera_optimizer.get_metrics_dict(metrics_dict)
        return metrics_dict

    def _get_image_loss(self, gt_img, pred_img, ssim_lambda=None):
        if ssim_lambda is None:
            ssim_lambda = self.config.ssim_lambda
        Ll1 = torch.abs(gt_img - pred_img).mean()
        simloss = 1 - self.ssim(gt_img.permute(2, 0, 1)[None, ...], pred_img.permute(2, 0, 1)[None, ...])
        return (1 - ssim_lambda) * Ll1 + ssim_lambda * simloss

    def _get_fft_magnitude_loss(self, gt_image, pred_image):
        gt_fft = torch.fft.fft2(gt_image.permute(2, 0, 1))
        pred_fft = torch.fft.fft2(pred_image.permute(2, 0, 1))
        gt_magnitude = torch.abs(gt_fft)
        pred_magnitude = torch.abs(pred_fft)
        _, H, W = gt_image.shape
        loss_magnitude = torch.abs(gt_magnitude - pred_magnitude).mean()
        return loss_magnitude

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        gt_img = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        num_levels = self.num_levels

        loss_dict = dict()
        gt_img = gt_img.permute(2, 0, 1).unsqueeze(0)
        ker = gauss_kernel(self.device, channels=3)
        for i in range(num_levels):
            cur_pred_gauss_level = outputs[f'gaussian_pyramid_{str(i)}']
            num_downsamples = num_levels - i - 1
            cur_pred_gauss_level_downsampled = downsample_antialias_n_times(cur_pred_gauss_level.permute(2, 0, 1).unsqueeze(0),
                                                                            ker, num_downsamples)
            cur_gt_downsampled = downsample_antialias_n_times(gt_img, ker, num_downsamples)
            lm_gauss = self.config.gauss_loss_lambda
            if i < num_levels - 1:
                lm_gauss *= self.config.lower_gauss_loss_lambda_factor
            cur_pred_gauss_level_downsampled = cur_pred_gauss_level_downsampled.squeeze(0).permute(1, 2, 0)
            cur_gt_downsampled = cur_gt_downsampled.squeeze(0).permute(1, 2, 0)
            loss_dict[f"gaussian_loss_{str(i)}"] = self._get_image_loss(cur_gt_downsampled, cur_pred_gauss_level_downsampled) * lm_gauss

            if i < num_levels - 1:
                cur_gt_blurred = downsample_n_times_upsample_n_times(gt_img, num_downsamples).squeeze(0).permute(1, 2, 0)
                cur_pred_gauss_level = cur_pred_gauss_level[:cur_gt_blurred.shape[0], :cur_gt_blurred.shape[1]]
                loss_dict[f"gaussian_loss_{str(i)}_fft"] = self._get_fft_magnitude_loss(cur_gt_blurred,
                                                                                        cur_pred_gauss_level) * self.config.gauss_loss_fft_lambda

        assert "mask" not in batch, "masking is not supported at the moment"

        if self.config.use_scale_regularization and self.step % 10 == 0:
            scale_exp = torch.exp(self.scales)
            scale_reg = (
                torch.maximum(
                    scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1),
                    torch.tensor(self.config.max_gauss_ratio),
                )
                - self.config.max_gauss_ratio
            )
            scale_reg = 0.1 * scale_reg.mean()
        else:
            scale_reg = torch.tensor(0.0).to(self.device)

        loss_dict["scale_reg"] = scale_reg

        if self.training:
            # Add loss from camera optimizer
            self.camera_optimizer.get_loss_dict(loss_dict)
            if self.config.use_bilateral_grid:
                loss_dict["tv_loss"] = 10 * total_variation_loss(self.bil_grids.grids)

        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Writes the test image outputs.

        Args:
            image_idx: Index of the image.
            step: Current step.
            batch: Batch of data.
            outputs: Outputs of the model.

        Returns:
            A dictionary of metrics.
        """
        cc_rgb = None
        gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        predicted_rgb = outputs["rgb"]
        assert abs(gt_rgb.shape[0] - predicted_rgb.shape[0]) < 10
        assert abs(gt_rgb.shape[1] - predicted_rgb.shape[1]) < 10
        gt_rgb = gt_rgb[:predicted_rgb.shape[0], :predicted_rgb.shape[1], :]

        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)

        if self.config.color_corrected_metrics:
            cc_rgb = color_correct(predicted_rgb, gt_rgb)
            cc_rgb = torch.moveaxis(cc_rgb, -1, 0)[None, ...]

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        if self.config.color_corrected_metrics:
            assert cc_rgb is not None
            cc_psnr = self.psnr(gt_rgb, cc_rgb)
            cc_ssim = self.ssim(gt_rgb, cc_rgb)
            cc_lpips = self.lpips(gt_rgb, cc_rgb)
            metrics_dict["cc_psnr"] = float(cc_psnr.item())
            metrics_dict["cc_ssim"] = float(cc_ssim)
            metrics_dict["cc_lpips"] = float(cc_lpips)

        images_dict = {"img": combined_rgb}

        return metrics_dict, images_dict
