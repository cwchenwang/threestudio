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

        x, eps = (x - sigma * noise_pred) / alpha, noise_pred
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
        rgb: Float[Tensor, "B H W C"],
        diff_rgbs: Float[Tensor, "B C H W"],
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
        loss_sds = F.mse_loss(latents, diff_rgbs) / batch_size
        # with torch.no_grad():
        #     fake_imgs = self.decode_latents(diff_rgbs)
        # loss_sds = F.mse_loss(rgb_BCHW_512, fake_imgs) / batch_size
        guidance_out = {
            "loss_sds": loss_sds,
        }
        return guidance_out
