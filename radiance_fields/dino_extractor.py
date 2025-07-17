# Author: Qian Liu
# Email: liu.qian.pro@gmail.com

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel

class DINOVitExtractor(nn.Module):
    def __init__(self, model_name, device, usev2=True):
        super().__init__()
        
        if usev2:
            # self.model = torch.hub.load('facebookresearch/dinov2', model_name,
            #     force_reload=True).to(device)
            self.model = ViTModel.from_pretrained(f'facebook/{model_name}', force_download=True,
                use_safetensors=True).to(device)
        else:
            self.model = torch.hub.load('facebookresearch/dino:main', model_name,
                force_reload=True).to(device)

        # self.model = torch.load('model/dino_vitbase8_pretrain.pth').to(device)  
        self.model.eval()
        self.last_block = None
        self.feature_output = None
        self.last_block = self.model.blocks[-1]
        self.last_block.register_forward_hook(self._get_block_hook())

    def _get_block_hook(self):
        def _get_block_output(model, input, output):
            self.feature_output = output
        return _get_block_output
    
    def get_vit_feature(self, input_img):
        mean = torch.tensor([0.485, 0.456, 0.406], device=input_img.device).reshape(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=input_img.device).reshape(1, 3, 1, 1)
        input_img = (input_img - mean) / std
        self.model(input_img)
        return self.feature_output