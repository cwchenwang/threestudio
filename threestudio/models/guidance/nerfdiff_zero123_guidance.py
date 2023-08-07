import importlib
import os
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler, StableDiffusionPipeline
from diffusers.utils.import_utils import is_xformers_available
from omegaconf import OmegaConf
from tqdm import tqdm

import threestudio
from threestudio.models.guidance.zero123_guidance import Zero123Guidance
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, parse_version
from threestudio.utils.typing import *


@threestudio.register("nerfdiff-zero123-guidance")
class NeRFDiffZero123Guidance(Zero123Guidance):
    @dataclass
    class Config(Zero123Guidance.Config):
        pass

    cfg: Config

    def configure(self):
        super().configure()

    @torch.no_grad()
    def diffusion_step(
        self,
        j,
        t,
        nerf_rgb,
        diff_misc,
        diff_epss,
        elevation,
        azimuth,
        camera_distances,
        first_iter=False,
    ):
        s = 1

        if first_iter:
            x, eps = nerf_rgb, torch.randn_like(nerf_rgb)
        else:
            tm1, alpha, sigma, z = diff_misc[j]
            eps = (z - alpha * nerf_rgb) / sigma
            eps = (eps - diff_epss[j]) * s + diff_epss[j]
            x = (z - sigma * eps) / alpha

        # core diffusion step
        alpha = self.alphas[t] ** 0.5
        sigma = (1 - self.alphas[t]) ** 0.5
        # t = t * eps.new_ones(eps.size(0))
        t = torch.tensor([t], dtype=torch.long, device=self.device)

        # alpha, sigma = t_to_alpha_sigma(ts)
        # z = x * alpha + eps * sigma
        # diff_output = model.forward_denoising(z, ts * 1000, *pre_rendered_otuputs[view_id])

        # add noise
        z = x * alpha + eps * sigma
        # pred noise
        x_in = torch.cat([z] * 2)
        t_in = torch.cat([t] * 2)
        cond = self.get_cond(elevation, azimuth, camera_distances)
        noise_pred = self.model.apply_model(x_in, t_in, cond)

        # perform guidance
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
            noise_pred_cond - noise_pred_uncond
        )

        x, eps = (z - sigma * noise_pred) / alpha, noise_pred
        # x = x.clamp(-1, 1)  # clip the values to avoid large numbers
        return {
            "x": x,
            "eps": eps,
            "ts": t.detach(),
            "alpha": alpha,
            "sigma": sigma,
            "z": z,
        }

    def __call__(
        self,
        t,
        idx,
        diff_misc,
        diff_epss,
        rgb: Float[Tensor, "B H W C"],
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        rgb_as_latents=False,
        guidance_eval=False,
        **kwargs,
    ):
        batch_size = rgb.shape[0]
        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        rgb_BCHW_512 = F.interpolate(
            rgb_BCHW, (256, 256), mode="bilinear", align_corners=False
        )
        latents = self.encode_images(rgb_BCHW_512)

        with torch.no_grad():
            x, eps = [], []
            s = 1
            for i, j in enumerate(idx):
                x_ = latents[i : i + 1]
                if diff_misc["first"][j]:
                    x_, eps_ = latents[i : i + 1], torch.randn_like(latents[i : i + 1])
                    diff_misc["first"][j] = False
                else:
                    alpha, sigma, z = (
                        diff_misc["alpha"][j],
                        diff_misc["sigma"][j],
                        diff_misc["z"][j],
                    )
                    eps_ = (z - alpha * latents[i : i + 1]) / sigma
                    eps_ = (eps_ - diff_epss[j]) * s + diff_epss[j]
                    x_ = (z - sigma * eps_) / alpha
                x.append(x_)
                eps.append(eps_)
            x = torch.cat(x, dim=0)
            eps = torch.cat(eps, dim=0)

        cond = self.get_cond(elevation, azimuth, camera_distances)
        # core diffusion step
        t = torch.tensor([t] * batch_size, dtype=torch.long, device=self.device)
        alpha = (self.alphas[t] ** 0.5).reshape(-1, 1, 1, 1)
        sigma = ((1 - self.alphas[t]) ** 0.5).reshape(-1, 1, 1, 1)

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # eps = torch.randn_like(x)
            # add noise
            z = x * alpha + eps * sigma
            # pred noise
            x_in = torch.cat([z] * 2)
            t_in = torch.cat([t] * 2)
            noise_pred = self.model.apply_model(x_in, t_in, cond)

        # perform guidance
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
            noise_pred_cond - noise_pred_uncond
        )

        # NeRFDiff Loss
        x_diff, eps = (z - sigma * noise_pred) / alpha, noise_pred
        loss_sds = 0.5 * F.mse_loss(latents, x_diff) / batch_size

        # # SDS Loss
        # w = (1 - self.alphas[t]).reshape(-1, 1, 1, 1)
        # grad = w * (noise_pred - eps)
        # grad = torch.nan_to_num(grad)
        # # clip grad for stable training?
        # if self.grad_clip_val is not None:
        #     grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)

        # # loss = SpecifyGradient.apply(latents, grad)
        # # SpecifyGradient is not straghtforward, use a reparameterization trick instead
        # target = (latents - grad).detach()
        # # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        # loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size

        guidance_out = {
            "loss_sds": loss_sds,
            # "x_diff": x_diff.detach(),
            "eps": eps.detach(),
            "alpha": alpha.detach(),
            "sigma": sigma.detach(),
            "z": z.detach(),
        }
        return guidance_out
