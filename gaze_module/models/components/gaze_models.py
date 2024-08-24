from functools import partial
import einops
from requests import get
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
from torchvision.models import get_model

class ResNet(nn.Module):

    def __init__(
        self,
        model_name: str,
        pretrained: bool,
        head_size: int
        ):
        super(ResNet, self).__init__()

        if model_name == "resnet18":
            self.model = resnet18(pretrained=pretrained)
            self.model.fc1 = nn.Linear(512, 768)
        # elif model_name == "resnet34":
        #     self.model = resnet34(pretrained=pretrained)
        elif model_name == "resnet50":
            self.model = resnet50(pretrained=pretrained)
            self.model.fc1 = nn.Linear(2048, 768)
        # elif model_name == "resnet101":
        #     self.model = resnet101(pretrained=pretrained)
        # elif model_name == "resnet152":
        #     self.model = resnet152(pretrained=pretrained)
        else:
            raise ValueError(f"Invalid model name: {model_name}")

        self.model.fc2 = nn.Identity()
        # output size 1000
        self.output_size = 768

    def forward(self, x):
        return self.model(x)

class TemporalEncoder(nn.Module):

    def __init__(self
                 ,input_size
        ):
        super(TemporalEncoder, self).__init__()
        self.input_size = input_size
        self.img_feature_dim = 256  # the dimension of the CNN feature to represent each frame

        # The LSTM layer
        self.lstm = nn.LSTM(self.img_feature_dim, self.img_feature_dim,bidirectional=True,num_layers=2,batch_first=True)

    def forward(self, input):

        b,t,d = input.size()

        
        base_out = base_out.view(input.size(0),7,self.img_feature_dim)

        lstm_out, _ = self.lstm(base_out)
        lstm_out = lstm_out[:,3,:]
        output = self.last_layer(lstm_out).view(-1,3)


        angular_output = output[:,:2]
        angular_output[:,0:1] = math.pi*nn.Tanh()(angular_output[:,0:1])
        angular_output[:,1:2] = (math.pi/2)*nn.Tanh()(angular_output[:,1:2])

        var = math.pi*nn.Sigmoid()(output[:,2:3])
        var = var.view(-1,1).expand(var.size(0),2)

        return angular_output,var

class TorchvisionEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        pretrained: bool,
        head_size: int
        ):
        super(TorchvisionEncoder, self).__init__()

        if model_name == "swin_v2_t":
            if pretrained:
                self.model = get_model("swin_v2_t", weights="DEFAULT")
            else:
                self.model = get_model("swin_v2_t")
            self.model.head = nn.Linear(768, 768)
            self.output_size = 768

        elif model_name == "inception_v3":
            if pretrained:
                self.model = get_model("inception_v3", weights="DEFAULT")
            else:
                self.model = get_model("inception_v3")
            self.model.fc = nn.Linear(2048, 768)
            self.model.aux_logits = False
            self.output_size = 768
        
        elif model_name == "omnivore":
            """ model that can accept image and video inputs """
            omivore = torch.hub.load("facebookresearch/omnivore", model="omnivore_swinT")
            self.model = omivore.trunk
            self.output_size = 768

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

class HeadCartesianFromSpherical(nn.Module):

    def __init__(self, in_features):
        super(HeadCartesianFromSpherical, self).__init__()

        self.head = nn.Linear(in_features, 3)
        self.softplus = nn.Softplus()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.head(x)
        angular_output = x[:, :2]
        angular_output[:, 0:1] = math.pi * self.tanh(angular_output[:, 0:1])
        angular_output[:, 1:2] = (math.pi / 2) * self.tanh(angular_output[:, 1:2])

        x = torch.cos(angular_output[:, 1:2]) * torch.sin(angular_output[:, 0:1])
        y = torch.sin(angular_output[:, 1:2])
        z = -torch.cos(angular_output[:, 1:2]) * torch.cos(angular_output[:, 0:1])
        
        var = self.softplus(x[:, 2:3])
        output = { 
            "cartesian" : torch.cat([x, y, z], dim=1),
            "var" : var
        } 
        return output

class HeadCartesian(nn.Module):

    def __init__(self, in_features):
        super(HeadCartesian, self).__init__()

        self.head_dir = nn.Linear(in_features, 3)
        self.head_var = nn.Linear(in_features, 1)
        self.softplus = nn.Softplus()

    def forward(self, x):
        dir = self.head_dir(x)
        var = self.softplus(self.head_var(x))
        output = { 
            "cartesian" : dir,
            "var" : var
        } 
        return output

class HeadSpherical(nn.Module):

    def __init__(self, in_features):
        super(HeadSpherical, self).__init__()

        self.head = nn.Linear(in_features, 3)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.head(x)
        angular_output = x[:, :2]
        angular_output[:, 0:1] = math.pi * self.tanh(angular_output[:, 0:1])
        angular_output[:, 1:2] = (math.pi / 2) * self.tanh(angular_output[:, 1:2])

        var = math.pi * nn.Sigmoid()(x[:, 2:3])
        output = { 
            "spherical" : angular_output,
            "var" : var
        } 
        return output

# generic model for gaze estimation 
class GazeNet(nn.Module):

    def __init__(
        self, 
        encoder: nn.Module,
        head : partial,
        activation: nn.Module = nn.Identity(),
        ):
        super(GazeNet, self).__init__()
        self.encoder = encoder
        self.head = head(in_features= encoder.output_size)
        self.activation = activation

    def forward(self, x, data_id):
        # x => B, T, C, H, W
        assert x.dim() == 5
        x = einops.rearrange(x, 'b t c h w -> b c t h w')
        if x.size(2) == 1:
            x = x.squeeze(2)
        
        x = self.encoder(x)
        x = self.activation(x)
        
        x_dict = self.head(x)

        return x_dict

# Baseline model for gaze estimation 
class BaseGazeNet(nn.Module):

    def __init__(
        self, 
        encoder: nn.Module,
        head : partial,
        activation: nn.Module = nn.Identity(),
        ):
        super(GazeNet, self).__init__()
        self.encoder = encoder
        self.head = head(in_features= encoder.output_size)
        self.activation = activation

    def forward(self, x, data_id):
        # x => B, T, C, H, W
        assert x.dim() == 5
        x = einops.rearrange(x, 'b t c h w -> b c t h w')
        if x.size(2) == 1:
            x = x.squeeze(2)
        
        x = self.encoder(x)
        x = self.activation(x)
        
        x_dict = self.head(x)

        return x_dict

    
