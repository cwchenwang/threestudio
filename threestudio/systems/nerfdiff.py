import os
from dataclasses import dataclass, field

import torch

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.misc import cleanup, get_device
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.perceptual import PerceptualLoss
from threestudio.utils.typing import *


@threestudio.register("nerfdiff-system")
class NeRFDiff(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        per_alternate_step: int = 10
        start_alternate_step: int = 1000
        batch_render: int = 3
        batch_guide: int = 4

    cfg: Config

    def configure(self):
        # create geometry, material, background, renderer
        super().configure()
        self.guidance_imgs = []
        self.guidance_imgs_info = {} # rays etc.
        self.perceptual_loss = PerceptualLoss().eval().to(get_device())
        self.start_alternate_step = self.cfg.start_alternate_step
        self.per_alternate_step = self.cfg.per_alternate_step
        self.batch_render = self.cfg.batch_render

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        render_out = self.renderer(**batch)
        return {
            **render_out,
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()
        # no prompt processor
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        self.input_image = self.trainer.datamodule.train_dataloader().dataset.get_all_images()
        self.save_image_grid(
            "all_training_images.png",
            [
                {"type": "rgb", "img": image.permute(2,0,1), "kwargs": {"data_format": "CHW"}} for image in self.input_image
            ],
            name="on_fit_start",
            step=self.true_global_step,
        )

    def training_step(self, batch, batch_idx):
        if self.global_step == 0 or (self.global_step > self.start_alternate_step and self.global_step % self.per_alternate_step == 0): 
            # update guidance images
            if self.global_step == 0:
                self.guidance_imgs = self.guidance.generate(self.input_image.permute(0,3,1,2), batch['random_camera']['elevation'], batch['random_camera']['azimuth'], batch['random_camera']['camera_distances'], stop_at=30)
                self.guidance_imgs_info = batch['random_camera']
            else: # render images from nerf
                img_render = []
                for i in range(self.guidance_imgs.shape[0] // self.cfg.batch_render):
                    data = {'render_full': True}
                    for key in self.guidance_imgs_info.keys():
                        data[key] = self.guidance_imgs_info[key][i*self.cfg.batch_render: (i+1)*self.cfg.batch_render].to(self.device) if isinstance(self.guidance_imgs_info[key], torch.Tensor) else self.guidance_imgs_info[key]
                    torch.cuda.empty_cache()
                    with torch.no_grad():
                        out = self(data)
                        img_render.append(out["comp_rgb"])
                        print(out["comp_rgb"].shape)
                img_render = torch.cat(img_render, dim=0)
                # TODO: update diffusion guidance
            self.save_image_grid(
                f"guidance_images-{self.true_global_step}.png",
                [
                    {"type": "rgb", "img": image, "kwargs": {"data_format": "HWC"}} for image in self.guidance_imgs
                ],
                name="guidance",
                step=self.true_global_step,
            )
            # self.save_data(f'guidance-data-{self.global_step}.npz', {'img': self.guidance_imgs, 'info': self.guidance_imgs_info})
            torch.cuda.empty_cache()

        idx = torch.randperm(self.guidance_imgs.shape[0])[:self.cfg.batch_guide]
        batch_data = { }
        batch_data["gt_rgb"] = torch.from_numpy(self.guidance_imgs[idx]).to(self.device)
        for key in self.guidance_imgs_info.keys():
            batch_data[key] = self.guidance_imgs_info[key][idx].to(self.device) if isinstance(self.guidance_imgs_info[key], torch.Tensor) else self.guidance_imgs_info[key]
        out = self(batch_data)

        loss = 0.0
        guidance_out = {
            "loss_l1": torch.nn.functional.l1_loss(out["comp_rgb"], batch_data["gt_rgb"]),
            "loss_p": self.perceptual_loss(
                out["comp_rgb"].permute(0, 3, 1, 2).contiguous(),
                batch_data["gt_rgb"].permute(0, 3, 1, 2).contiguous(),
            ).sum(),
        }

        for name, value in guidance_out.items():
            self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

        if self.C(self.cfg.loss.lambda_orient) > 0:
            if "normal" not in out:
                raise ValueError(
                    "Normal is required for orientation loss, no normal is found in the output."
                )
            loss_orient = (
                out["weights"].detach()
                * dot(out["normal"], out["t_dirs"]).clamp_min(0.0) ** 2
            ).sum() / (out["opacity"] > 0).sum()
            self.log("train/loss_orient", loss_orient)
            loss += loss_orient * self.C(self.cfg.loss.lambda_orient)

        loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
        self.log("train/loss_sparsity", loss_sparsity)
        loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

        opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
        loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
        self.log("train/loss_opaque", loss_opaque)
        loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0]}.png",
            [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            name="validation_step",
            step=self.true_global_step,
        )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-test/{batch['index'][0]}.png",
            [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            name="test_step",
            step=self.true_global_step,
        )

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )
