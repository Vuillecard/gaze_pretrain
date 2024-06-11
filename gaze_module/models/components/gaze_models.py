import torch 
from torch import nn
from gaze_module.models.resnets.resnet import (
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152
)
import math

class ResNet(nn.Module):

    def __init__(
        self,
        model_name: str,
        pretrained: bool
        ):
        super(ResNet, self).__init__()

        if model_name == "resnet18":
            self.model = resnet18(pretrained=pretrained)
        elif model_name == "resnet34":
            self.model = resnet34(pretrained=pretrained)
        elif model_name == "resnet50":
            self.model = resnet50(pretrained=pretrained)
        elif model_name == "resnet101":
            self.model = resnet101(pretrained=pretrained)
        elif model_name == "resnet152":
            self.model = resnet152(pretrained=pretrained)
        else:
            raise ValueError(f"Invalid model name: {model_name}")

        self.model.fc2 = nn.Identity()
        # output size 1000
        self.output_size = 1000

    def forward(self, x):
        return self.model(x)
    

class LinearHead(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearHead, self).__init__()

        self.head = nn.Linear(in_features, out_features)

    def forward(self, x):

        x = self.head(x)
        return x

    @classmethod
    def from_config(cls, config):
        return cls(
            in_features=config.in_features,
            out_features=config.out_features
        )

class LinearHeadSpherical(nn.Module):
    def __init__(self, in_features):
        super(LinearHeadSpherical, self).__init__()

        self.head = nn.Linear(in_features, 3)

    def forward(self, x):

        x = self.head(x)
      
        angular_output = x[:, :2]
        angular_output[:, 0:1] = math.pi * nn.Tanh()(angular_output[:, 0:1])
        angular_output[:, 1:2] = (math.pi / 2) * nn.Tanh()(angular_output[:, 1:2])

        var = math.pi * nn.Sigmoid()(x[:, 2:3])
        #var = var.view(-1, 1).expand(var.size(0), 2)

        output = torch.cat([angular_output, var], dim=1)
        return output

class LinearHeadCartesian(nn.Module):
    def __init__(self, in_features):
        super(LinearHeadCartesian, self).__init__()

        self.head = nn.Linear(in_features, 3)

        
    def forward(self, x):

        x = self.head(x)
      
        angular_output = x[:, :2]
        angular_output[:, 0:1] = math.pi * nn.Tanh()(angular_output[:, 0:1])
        angular_output[:, 1:2] = (math.pi / 2) * nn.Tanh()(angular_output[:, 1:2])

        var = math.pi * nn.Sigmoid()(x[:, 2:3])
        #var = var.view(-1, 1).expand(var.size(0), 2)

        x = torch.cos(angular_output[:, 1:2]) * torch.sin(angular_output[:, 0:1])
        y = torch.sin(angular_output[:, 1:2])
        z = -torch.cos(angular_output[:, 1:2]) * torch.cos(angular_output[:, 0:1])
        
        output = torch.cat([x, y, z, var], dim=1)

        return output
       

# generic model for gaze estimation 
class GazeNet(nn.Module):

    def __init__(
        self, 
        encoder: nn.Module,
        head : str,
        mode: str
        ):
        super(GazeNet, self).__init__()

        self.encoder = encoder 

        if head == "linear":
            if mode == "cartesian":
                self.head = LinearHead(
                    in_features=self.encoder.output_size,
                    out_features=4
                )
            elif mode == "spherical":
                self.head = LinearHeadSpherical(
                    in_features=self.encoder.output_size
                )
            elif mode == "spherical_to_cartesian":
                self.head = LinearHeadCartesian(
                    in_features=self.encoder.output_size
                )
            else:
                raise ValueError(f"Invalid mode: {mode}")
        else: 
            raise ValueError(f"Invalid head: {head}")

    def forward(self, x, data_id):
        x = x.squeeze(1)
        x = self.encoder(x)
        x = self.head(x)

        return x    
    