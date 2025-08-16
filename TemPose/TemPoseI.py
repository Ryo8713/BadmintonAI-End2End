import numpy as np
import torch
from torch import nn, einsum, Tensor
import torch.nn.functional as F
from einops import rearrange, repeat
from TemPose.NN_models import TCN, TransformerEncoder, MLP_Head
from TemPose.utility_tempose import get_2d_sincos_pos_embed
from positional_encodings.torch_encodings import PositionalEncoding1D
from torchinfo import summary

class TemPose_TF(nn.Module):
    '''Similar to TemPose_TF in TemPose.'''
    def __init__(
        self, in_dim, seq_len, n_class=35, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5
    ):
        '''`d_model` should be an even number.'''
        super().__init__()
        if n_people > 2:
            raise NotImplementedError

        self.project = nn.Linear(in_dim, d_model)

        # TCNs
        tcn_channels = [d_model // 2, d_model]
        self.tcn_top = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn_bottom = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn_shuttle = TCN(2, tcn_channels, tcn_kernel_size, drop_p)

        # Temporal TransformerLayers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, n_people+3, 1+seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # Interactional TransformerLayers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1+n_people+3, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)

        # MLP Head
        self.mlp_head = MLP_Head(d_model, n_class, d_model * mlp_d_scale, drop_p)

        self.d_model = d_model

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        # Positional encodings are different from TemPose.
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem.squeeze(0))
        self.embedding_tem.copy_(pos_encoding.unsqueeze(0))

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        # Same as TemPose here.
        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        # Same as TemPose
        if isinstance(m, nn.Linear):
            # following official JAX ViT xavier.uniform is used:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(
        self,
        JnB: Tensor,  # JnB: (b, t, n, input_dim)
        pos: Tensor,  # pos: (b, t, n, 2)
        shuttle: Tensor,  # shuttle: (b, t, 2)
        video_len: Tensor  # video_len: (b)
    ):
        JnB = JnB.transpose(1, 2).contiguous()
        # JnB: (b, n, t, input_dim)
        
        x = self.project(JnB)
        b, n, t, d = x.shape

        pos_top = pos[:, :, 0, :].transpose(1, 2).contiguous()
        pos_bottom = pos[:, :, 1, :].transpose(1, 2).contiguous()
        shuttle = shuttle.transpose(1, 2).contiguous()
        # pos_top: (b, 2, t)
        # pos_bottom: (b, 2, t)
        # shuttle: (b, 2, t)

        # TCNs
        pos_top: Tensor = self.tcn_top(pos_top)
        pos_bottom: Tensor = self.tcn_bottom(pos_bottom)
        shuttle: Tensor = self.tcn_shuttle(shuttle)
        # pos_top: (b, d, t)
        # pos_bottom: (b, d, t)
        # shuttle: (b, d, t)

        pos_top = pos_top.transpose(1, 2)
        pos_bottom = pos_bottom.transpose(1, 2)
        shuttle = shuttle.transpose(1, 2)
        x_additional = torch.stack((pos_top, pos_bottom, shuttle), dim=1)
        # x_additional: (b, 3, t, d)

        # Temporal Fusion (TF)
        x = torch.cat((x, x_additional), dim=1)
        n += 3

        # Concat cls token (temporal)
        class_token_tem = self.learned_token_tem.view(1, 1, 1, -1).expand(b, n, -1, -1)
        x = torch.cat((class_token_tem, x), dim=2)
        t += 1

        # Temporal embedding
        x = x + self.embedding_tem
        x: Tensor = self.pre_dropout(x)

        # Temporal TransformerLayers
        x = x.view(b*n, t, d)

        range_t = torch.arange(0, t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        # mask: (b, t)
        mask = mask.repeat_interleave(n, dim=0)
        # mask: (b*n, t)
        
        x = self.encoder_tem(x, mask)
        x = x[:, 0].view(b, n, d)

        # Concat cls token (interactional)
        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        x = torch.cat((class_token_inter, x), dim=1)
        n += 1

        # Interactional embedding
        x = x + self.embedding_inter

        # Interactional TransformerLayers
        x = self.encoder_inter(x)
        x = x[:, 0].contiguous()

        x = self.mlp_head(x)
        return x