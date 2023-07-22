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
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, parse_version
from threestudio.utils.typing import *
from threestudio.models.guidance.zero123_guidance import Zero123Guidance

@threestudio.register("nerfdiff-zero123-guidance")
class NeRFDiffZero123Guidance(Zero123Guidance):
    @dataclass
    class Config(Zero123Guidance.Config):
        pass

    cfg: Config

    def configure(self):
        super().configure()
    
    def __call__(
        self
    ):
        pass