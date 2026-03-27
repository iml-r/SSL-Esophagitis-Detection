import requests
import os

import timm
import torch

from typing import Literal

from timm.models import load_pretrained

class EsophagitisFeatureExtractor(torch.nn.Module):
    def __init__(self,
                 size: Literal["small", "base", "large"],
                 model_dir_path="./SSL-Esophagitis-Detection/downloaded_models",
                 pretrained_on_esophagitis=True,
                 use_qkvb = True,
                 use_global_pool = True,
                 *args, **kwargs):
        super().__init__()

        self.vision_model_size = size
        self.vision_model_name = f'vit_{self.vision_model_size}_patch16_dinov3'
        self.pretrained_on_esophagitis = pretrained_on_esophagitis

        if use_qkvb:
            self.vision_model_name = self.vision_model_name + "_qkvb"

        self.model_dir_path = model_dir_path
        self.model_path = f"{model_dir_path}/dinov3-vit{self.vision_model_size[0]}16-esophagitis-detector.pth"

        self.vision_model = timm.create_model(self.vision_model_name, pretrained=True, features_only=True)
        self.global_pool = torch.nn.AdaptiveAvgPool2d((1, 1)) if use_global_pool else None



        if self.pretrained_on_esophagitis:
            if not os.path.exists(self.model_path):
                print("Model not found locally. Downloading...")
                self.download_model()

            print(f"Loading EsophagitisFeatureExtractor model from {self.model_path}...")

            try:
                ckpt = torch.load(self.model_path, weights_only=False)

                # The state dict keys in the checkpoint are expected to be prefixed with "model.",
                # so we add that prefix to each key before loading.
                state_dict = {"model."+k : v for k, v in ckpt.items()}

                self.vision_model.load_state_dict(state_dict, strict=False)

            except Exception as e:
                raise Exception("Error loading state dict with strict=True. Reason:", e)

    def forward(self, x,
                *args, **kwargs):
        output = self.vision_model(x)

        if self.global_pool is not None:
            output = self.global_pool(output[-1])  # Apply global pooling to the last feature map

        return output

    def download_model(self):
        link = f"https://huggingface.co/tofriede/dinov3-upperGI/resolve/main/dinov3-vit{self.vision_model_size[0]}16-pretrain-upperGI400k.pth"

        # Download the model file
        response = requests.get(link)

        if response.status_code == 200:
            os.makedirs(self.model_dir_path, exist_ok=True)

            with open(self.model_path, 'wb') as f:
                f.write(response.content)

            print(f"EsophagitisFeatureExtractor model of size {self.vision_model_size} downloaded successfully.")

        else:
            raise Exception("Failed to download the model. Status code:", response.status_code)
