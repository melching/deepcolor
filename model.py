import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import timm


class BasicModel(nn.Module):
    
    def __init__(self, img_size):
        super().__init__()
        
        self.img_x = img_size[0]
        self.img_y = img_size[1]
        
    def forward(self, x):
        return x

     
class BasicTimmModel(nn.Module):
    
    def __init__(self, timm_name, img_size, pretrained=True):
        super().__init__()
        
        self.img_x = img_size[0]
        self.img_y = img_size[1]
        
        self.backbone = timm.create_model(timm_name, pretrained=pretrained, num_classes=0)
        
    def forward(self, x):
        x = self.backbone(x)
        return x
    
class Unet(nn.Module):

    def __init__(self, img_size, in_channels=1, encoder="mobilenet_v2", layernorm=True, activation=None):
        super().__init__()
        
        assert activation in [None, "sigmoid"]
        
        self.img_x = img_size[0]
        self.img_y = img_size[1]
        
        self.backbone = smp.Unet(
            encoder_name=encoder,
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=3,
        )
        
        self.layernorm = None
        if layernorm:
            self.layernorm = nn.LayerNorm([3, self.img_x, self.img_y])
        
        self.activation = None
        if activation == "sigmoid":
            self.activation = nn.Sigmoid()
        
    def forward(self, x):
        x = self.backbone(x)
        if self.layernorm:
            x = self.layernorm(x)
        if self.activation:
            x = self.activation(x)
        return x