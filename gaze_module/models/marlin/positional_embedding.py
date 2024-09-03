import torch
from torch import Tensor, nn
from einops import rearrange
from .modules import Shape


class PositionalEmbedding(nn.Module):
    def __init__(self, input_shape: Shape, dropout_rate: float = 0.5, trainable: bool = True):
        super().__init__()
        self.input_shape = input_shape
        self.emb = nn.Parameter(torch.zeros(1, *input_shape), requires_grad=trainable)
        self.use_dropout = dropout_rate is not None and dropout_rate != 0.0
        if self.use_dropout:
            self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.emb
        if self.use_dropout:
            x = self.dropout(x)
        return x

    @property
    def trainable(self):
        return self.emb.requires_grad

    @trainable.setter
    def trainable(self, value: bool):
        self.emb.requires_grad = value


class SinCosPositionalEmbedding(PositionalEmbedding):
    def __init__(self, input_shape: Shape, dropout_rate: float = 0.5):
        super().__init__(input_shape, dropout_rate, trainable=False)
        self.emb.data = self.make_embedding().unsqueeze(0)

    def make_embedding(self) -> Tensor:
        n_position, d_hid = self.input_shape

        def get_position_angle_vec(position):
            return position / torch.tensor(10000).pow(
                2 * torch.div(torch.arange(d_hid), 2, rounding_mode="trunc") / d_hid
            )

        sinusoid_table = torch.stack(
            [get_position_angle_vec(pos_i) for pos_i in range(n_position)], 0
        )
        sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return sinusoid_table.float()
    
    def interpolate_embedding(self, n_position_new: int):

        pos_embedding = self.emb.data
        sincosembed = rearrange(pos_embedding, "b (nt nh nw) d -> b d nt nh nw", nt=8, nh=14, nw=14)
        nt_new ,nh_new, nw_new = int(n_position_new/(14*14)), 14,14
        sincosembed_new = torch.nn.functional.interpolate(
                    sincosembed, size=(nt_new ,nh_new, nw_new), mode='trilinear', align_corners=False)
        
        sincosembed_new = rearrange(sincosembed_new, "b d nt nh nw -> b (nt nh nw) d")
        self.emb.data = sincosembed_new.float()

class SinCosInterpolationPositionalEmbedding(PositionalEmbedding):
    def __init__(self, input_shape: Shape, dropout_rate: float = 0.5):
        super().__init__(input_shape, dropout_rate, trainable=False)
        self.emb.data = self.make_embedding()
        self.embedding_og = self.make_embedding().to(self.emb.device)

    def make_embedding(self) -> Tensor:
        n_position, d_hid = 1568, 384

        def get_position_angle_vec(position):
            return position / torch.tensor(10000).pow(
                2 * torch.div(torch.arange(d_hid), 2, rounding_mode="trunc") / d_hid
            )

        sinusoid_table = torch.stack(
            [get_position_angle_vec(pos_i) for pos_i in range(n_position)], 0
        )
        sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        sincosembed = sinusoid_table.unsqueeze(0)
        assert sincosembed.shape == (1, 1568, d_hid)
        return sincosembed.float()

    def interpolate_embedding(self, n_position_new: int):

        pos_embedding = self.embedding_og
        sincosembed = rearrange(pos_embedding, "b (nt nh nw) d -> b d nt nh nw", nt=8, nh=14, nw=14)
        nt_new ,nh_new, nw_new = int(n_position_new/(14*14)), 14,14
        sincosembed_new = torch.nn.functional.interpolate(
                    sincosembed, size=(nt_new ,nh_new, nw_new), mode='trilinear', align_corners=False)
        
        sincosembed_new = rearrange(sincosembed_new, "b d nt nh nw -> b (nt nh nw) d")
        self.emb.data = sincosembed_new.float()

    def forward(self, x: Tensor) -> Tensor:
        self.interpolate_embedding(x.size(1))

        x = x + self.emb
        if self.use_dropout:
            x = self.dropout(x)
        return x
    

class PositionalEmbeddingInterpolation(nn.Module):
    def __init__(self, input_shape: Shape, dropout_rate: float = 0.5):
        super().__init__()
        self.input_shape = input_shape
        
        self.emb = nn.Parameter(torch.zeros(1, 1568, 384), requires_grad=False)
       
        # init embeddings 
        self.emb.data = self.make_embedding(8)

        self.use_dropout = dropout_rate is not None and dropout_rate != 0.0
        if self.use_dropout:
            self.dropout = nn.Dropout(dropout_rate)

    def set_interpolation_emb(self):
        self.emb_3 = nn.Parameter(torch.zeros(1, 3*14*14,384), requires_grad=False)
        self.emb_15 = nn.Parameter(torch.zeros(1, 15*14*14,384), requires_grad=False)
        self.emb_3.data = self.make_embedding(3)
        self.emb_15.data = self.make_embedding(15)
    
    def make_embedding(self,n_t) -> Tensor:
        n_position, d_hid = 1568, 384

        def get_position_angle_vec(position):
            return position / torch.tensor(10000).pow(
                2 * torch.div(torch.arange(d_hid), 2, rounding_mode="trunc") / d_hid
            )

        sinusoid_table = torch.stack(
            [get_position_angle_vec(pos_i) for pos_i in range(n_position)], 0
        )
        sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        sinusoid_table = sinusoid_table.float().unsqueeze(0)

        sincosembed = rearrange(sinusoid_table, "b (nt nh nw) d -> b d nt nh nw", nt=8, nh=14, nw=14)
        nt_new ,nh_new, nw_new = int(n_t), 14,14
        sincosembed_new = torch.nn.functional.interpolate(
                    sincosembed, size=(nt_new ,nh_new, nw_new), mode='trilinear', align_corners=False)
        
        sincosembed_new = rearrange(sincosembed_new, "b d nt nh nw -> b (nt nh nw) d")
        
        return sincosembed_new
    
    def forward(self, x: Tensor) -> Tensor:
        n_tokens = x.size(1)
        nt = int(n_tokens/(14*14))
        print("nt",nt)
        if nt == 3:
            x = x + self.emb_3
        elif nt == 8:
            x = x + self.emb
        elif nt == 15:
            x = x + self.emb_15
        else:
            raise ValueError("Invalid number of tokens")
        
        if self.use_dropout:
            x = self.dropout(x)
        return x

    @property
    def trainable(self):
        return self.emb.requires_grad

    @trainable.setter
    def trainable(self, value: bool):
        self.emb.requires_grad = value