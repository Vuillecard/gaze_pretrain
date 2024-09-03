import os

import torch
from torch import Tensor, nn
from torch.nn import LayerNorm, ModuleList
from einops import repeat, rearrange
from src.models.marlin.config import resolve_config

from .modules import Block, PatchEmbedding3d
from .positional_embedding import SinCosPositionalEmbedding, SinCosInterpolationPositionalEmbedding,PositionalEmbeddingInterpolation
from .utils import trunc_normal_


class MarlinEncoder(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        n_frames=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        norm_layer="LayerNorm",
        init_values=0.0,
        tubelet_size=2,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.patch_embedding = PatchEmbedding3d(
            input_size=(3, n_frames, img_size, img_size),
            patch_size=(tubelet_size, patch_size, patch_size),
            embedding=embed_dim,
        )
        num_patches = (
            (img_size // patch_size) * (img_size // patch_size) * (n_frames // tubelet_size)
        )

        # sine-cosine positional embeddings
        #self.pos_embedding = SinCosPositionalEmbedding((num_patches, embed_dim), dropout_rate=0.0)
        #self.pos_embedding = SinCosPositionalEmbedding((1568, embed_dim), dropout_rate=0.0)
        # new interpolation function for new frame size
        self.pos_embedding = PositionalEmbeddingInterpolation((num_patches, embed_dim), dropout_rate=0.0)

        if norm_layer == "LayerNorm":
            self.norm_layer = LayerNorm
            self.norm = self.norm_layer(embed_dim)
        else:
            raise NotImplementedError("Only LayerNorm is supported")

        self.blocks = ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    norm_layer=self.norm_layer,
                    init_values=init_values,
                )
                for _ in range(depth)
            ]
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x

    def add_tasktoken(self, num_task :int ) -> None:
        self.num_tasks = num_task
        self.task_tokens = nn.Parameter(torch.zeros(1, num_task, self.embed_dim))
        trunc_normal_(self.task_tokens, std=0.02)
    
    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        # mask: (B, T, N) with boolean values, 0 -> masked, 1 -> visible
        assert len(x.shape) == 5, "x must be 5D"
        emb = self.patch_embedding(x)
        emb = self.pos_embedding(emb)
        b, _, c = emb.shape
        emb = emb[mask].view(b, -1, c)  # only visible patches are used
        emb = self.forward_features(emb)
        return emb

    def extract_features(self, x: Tensor) -> Tensor:
        b, c, t, h, w = x.shape
        x = self.patch_embedding(x)
        x = self.pos_embedding(x)
        
        # Add global task tokens to input token
        cls_token = repeat(self.task_tokens, "() n d -> b n d", b=b)
        x = torch.cat([cls_token, x], dim=1)
        
        for block in self.blocks:
            x = block(x)

        # note imporant to apply norm on the output => decoder
        return x 
    
        # if seq_mean_pool:
        #     x = rearrange(x,"b (nt nh nw) d -> b nt nh nw d", nh = 14, nw = 14)
        #     x = x[:,x.shape[1]//2]
        #     x = rearrange(x, "b nh nw d -> b (nh nw) d")
        #     x = x.mean(dim=1)
        
        #     return x
        
        # # need to keep the normalized layer here
        # x = self.norm(x)
        # return x[:,:self.num_task]

    @classmethod
    def load_pretrain_marlin(cls, model_name, **kwargs):
        models_path = {
            "marlin_vit_small_ytf": os.path.join(
                torch.hub.get_dir(), "marlin_vit_small_ytf.encoder.pt"
            )
        }

        assert model_name in models_path, f"Model {model_name} not available"

        state_dict = torch.load(models_path[model_name], map_location="cpu")
        for key in list(state_dict.keys()):
            state_dict[key.removeprefix("encoder.")] = state_dict.pop(key)

        config = resolve_config(model_name)
        print(config)
        # for key, value in config.__dict__.items():
        #     print(key, value)

        # for key, value in kwargs.items():
        #     setattr(config, key, value)
        
        # for key, value in config.__dict__.items():
        #     print(key, value)

        n_frames = kwargs.get("n_frames", config.n_frames) 
        print(n_frames)
        model = cls(
            img_size=config.img_size,
            patch_size=config.patch_size,
            n_frames=n_frames,
            embed_dim=config.encoder_embed_dim,
            depth=config.encoder_depth,
            num_heads=config.encoder_num_heads,
            mlp_ratio=config.mlp_ratio,
            qkv_bias=config.qkv_bias,
            qk_scale=config.qk_scale,
            drop_rate=config.drop_rate,
            attn_drop_rate=config.attn_drop_rate,
            norm_layer=config.norm_layer,
            init_values=config.init_values,
            tubelet_size=config.tubelet_size,
        )
        for k, v in model.__dict__.items():
            print(k, v)
        # print('#'*50)
        # for k,v in state_dict.items():
        #     print(k)

        # print('#'*50)
        # for name,param in model.named_parameters():
        #     print(name)

        model.load_state_dict(state_dict)
        if n_frames != 16:
            #model.pos_embedding.interpolate_embedding((n_frames//2)*14*14)
            pass
        model.pos_embedding.set_interpolation_emb()
        model.add_tasktoken(2)
        return model
