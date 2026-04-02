import requests
import os

import timm
import timm.data
import torch

from typing import Literal

from timm.models.eva import checkpoint_filter_fn


class EsophagitisFeatureExtractor(torch.nn.Module):
    def __init__(self,
                 size: Literal["small", "base", "large"],
                 model_dir_path="./SSL-Esophagitis-Detection/downloaded_models",
                 pretrained_on_esophagitis=True,
                 use_global_pool = True,
                 use_strict_weight_loading = True,
                 use_native_transform = True,
                 *args, **kwargs):
        super().__init__()

        self.vision_model_size = size
        self.use_native_transform = use_native_transform
        self.vision_model_dict = {
            "small": "vit_small_patch16_dinov3_qkvb.lvd1689m",
            "base": "vit_base_patch16_dinov3_qkvb.lvd1689m",
            "large": "vit_large_patch16_dinov3_qkvb.lvd1689m"
        }

        self.vision_model_name = self.vision_model_dict[size]
        self.pretrained_on_esophagitis = pretrained_on_esophagitis

        # if use_qkvb:
        #     self.vision_model_name = self.vision_model_name + "_qkvb"

        self.model_dir_path = model_dir_path
        self.model_path = f"{model_dir_path}/dinov3-vit{self.vision_model_size[0]}16-esophagitis-detector.pth"

        self.vision_model = timm.create_model(self.vision_model_name,
                                              pretrained=True,
                                              features_only=True,
                                              pretrained_cfg_overlay={
                                                  "file": self.model_path if pretrained_on_esophagitis else None,
                                              }
                                              )

        self.global_pool = torch.nn.AdaptiveAvgPool2d((1, 1)) if use_global_pool else None

        self.use_strict_weight_loading = use_strict_weight_loading

        data_config = timm.data.resolve_model_data_config(self.vision_model)
        self.transform = timm.data.create_transform(**data_config,
                                                    is_training=False)

    def forward(self, x,
                *args, **kwargs):

        if self.use_native_transform:
            transformed_output = self.transform(x)
        else:
            transformed_output = x

        output = self.vision_model(transformed_output)

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
