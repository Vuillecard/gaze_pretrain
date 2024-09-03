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
from gaze_module.models.marlin.marlin_net import MarlinEncoder
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

class MarlinPretrainEncoder(nn.Module):
    def __init__(self, model_name: str, n_frames: int = 16, out_dim: int = 768):
        super(MarlinPretrainEncoder, self).__init__()

        self.marlin = MarlinEncoder.load_pretrain_marlin(model_name, n_frames=n_frames)
        self.output_dim = self.marlin.embed_dim
        self.head_out = nn.Linear(self.output_dim, out_dim)
        # layer norm 
        self.norm = nn.LayerNorm(self.output_dim)
    def forward(self, x): 
        # should be even number of frames
        b, t, c, h, w = x.size()        
        assert t % 2 == 0, f"Expected even number of frames, got {t}"

        x = einops.rearrange(x, "b t c h w -> b c t h w")      
        x = self.marlin.extract_features(x )

        # x = einops.rearrange(x[:, 2:],"b (nt nh nw) d -> b nt (nh nw) d", nh = 14, nw = 14)
        # x = x[:, x.shape[1]//2] # b n d 
        # x = x.mean(dim=1) # b d
        
        x = x[:,2:].mean(dim=1) # b d

        x = self.norm(x)
        x = self.head_out(x)
        
        return x
    
class TemporalEncoder(nn.Module):

    def __init__(self
                 ,input_size
        ):
        super(TemporalEncoder, self).__init__()
        self.input_size = input_size
        self.img_feature_dim = 256  # the dimension of the CNN feature to represent each frame
        self.output_size = self.img_feature_dim # because of bidirectional LSTM

        # The LSTM layer
        self.linear = nn.Linear(self.input_size, self.img_feature_dim)
        self.lstm = nn.LSTM(self.img_feature_dim, self.img_feature_dim, bidirectional=True, num_layers=2, batch_first=True)
        self.head_out = nn.Linear(self.img_feature_dim*2, self.output_size)

    def forward(self, input):

        B,T,D = input.size()
        
        if D != self.img_feature_dim:
            # we need to first embed the input to the img_feature_dim
            input = einops.rearrange(input, 'b t d -> (b t) d')
            input = self.linear(input)
            input = einops.rearrange(input, '(b t) d -> b t d' , b=B, t=T)

        #base_out = base_out.view(input.size(0),7,self.img_feature_dim)
        lstm_out, _ = self.lstm(input)
        lstm_out = lstm_out[:,T//2,:]
        lstm_out = self.head_out(lstm_out)
        
        return lstm_out

class TorchvisionEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        pretrained: bool,
        head_size: int
        ):
        super(TorchvisionEncoder, self).__init__()
        self.model_name = model_name

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
        
        elif model_name == "resnet50":
            self.model = resnet50(pretrained=pretrained)
            self.model.fc1 = nn.Linear(2048, 768)
            self.model.fc2 = nn.Identity()
            self.output_size = 768
        
        elif model_name == "resnet18":
            self.model = resnet18(pretrained=pretrained)
            self.model.fc1 = nn.Linear(512, 768)
            self.model.fc2 = nn.Identity()
            self.output_size = 768
        
        elif model_name == "marlin":
            self.model = MarlinPretrainEncoder("marlin_vit_small_ytf", n_frames=6,out_dim=768)
            self.output_size = 768

        elif model_name =="videoSwin":
            self.model = get_model("swin3d_t", weights="DEFAULT")
            self.model.head = nn.Linear(self.model.num_features, 768)
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

class GazeNetVideo(nn.Module):

    def __init__(
        self, 
        encoder: nn.Module,
        head : partial,
        activation: nn.Module = nn.Identity(),
        rearrange: bool = False,
        ):
        super(GazeNetVideo, self).__init__()
        self.encoder = encoder
        self.head = head(in_features= encoder.output_size)
        self.activation = activation
        self.rearrange = rearrange

    def forward(self, x, data_id):
        # x => B, T, C, H, W
        B,T,C,H,W = x.size()
        assert x.dim() == 5
        #duplicate the last frame to have even temporal dimension
        if T == 1:
            # we duplicate the frame to have a temporal dimension
            # mainly for testing use case
            x = x.repeat(1, 7, 1, 1, 1)
            B,T,C,H,W = x.size()

        if T % 2 != 0:
            x = x[:,:-1]
            B,T,C,H,W = x.size()
        
        if self.rearrange:
            x = einops.rearrange(x, 'b t c h w -> b c t h w')
        
        x = self.encoder(x)
        x = self.activation(x)
        
        x_dict = self.head(x)

        return x_dict


class GazeNetTemporal(nn.Module):

    def __init__(
        self, 
        encoder: nn.Module,
        head : partial,
        temporal_encoder: partial,
        activation: nn.Module = nn.Identity(),
        
        ):
        super(GazeNetTemporal, self).__init__()
        self.encoder = encoder
        self.temporal_encoder = temporal_encoder(input_size=encoder.output_size)
        self.head = head(in_features= self.temporal_encoder.output_size)
        self.activation = activation

    def forward(self, x, data_id):
        # x => B, T, C, H, W
        B,T,C,H,W = x.size()
        assert x.dim() == 5
        if T == 1:
            # we duplicate the frame to have a temporal dimension
            # mainly for testing use case
            x = x.repeat(1, 7, 1, 1, 1)
            B,T,C,H,W = x.size()

        x = einops.rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.encoder(x)

        x = einops.rearrange(x, '(b t) d -> b t d' , b=B, t=T)
        x = self.temporal_encoder(x)

        x = self.activation(x)
        x_dict = self.head(x)

        return x_dict

class GazeNetBaseline(nn.Module):

    def __init__(
        self, 
        encoder: nn.Module,
        head : partial,
        temporal_encoder: partial,
        activation: nn.Module = nn.Identity(),
        shared : bool = False,
        ):
        super(GazeNetBaseline, self).__init__()
        self.encoder = encoder
        self.temporal_encoder = temporal_encoder(input_size=256)
        self.head = head(in_features= self.temporal_encoder.output_size)
        self.head_video = head(in_features= self.temporal_encoder.output_size)
        self.adapter = nn.Linear(encoder.output_size, 256)
        self.activation = activation
        self.shared = shared

    def forward(self, x, data_id):
        # x => B, T, C, H, W
        B,T,C,H,W = x.size()
        assert x.dim() == 5

        x = einops.rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.encoder(x)
        x = self.activation(x)
        x = self.adapter(x) # (b*t) x 256

        if T > 1:
            x = einops.rearrange(x, '(b t) d -> b t d' , b=B, t=T)
            x = self.temporal_encoder(x)

        # x is bxd=256
        if self.shared:
            x = self.activation(x)
            x_dict = self.head(x)
        else:
            if T == 1:
                x = self.activation(x)
                x_dict = self.head(x)
            else:
                x = self.activation(x)
                x_dict = self.head_video(x)

        return x_dict
    



    
