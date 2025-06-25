import numpy as np
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from TemPose.NN_models import PreNorm, FeedForward, Attention, Transformer
from TemPose.utility_tempose import get_2d_sincos_pos_embed

class TemPoseII_TF(nn.Module):
    def __init__(self, *, poses_numbers, time_steps, num_people, num_classes, dim=50, kernel_size=5, depth=4, depth_int=3,
                 heads=6, scale_dim=4, mlp_dim=512, pool='cls', dim_head=75, dropout=0.3, emb_dropout=0.3):
        super().__init__()

        self.heads = heads
        self.time_sequence = time_steps
        self.people = num_people + 3  # 2 players + 1 shuttle
        self.to_patch_embedding = nn.Linear(poses_numbers, dim)

        self.temporal_embedding = nn.Parameter(torch.zeros(1, self.people, time_steps + 1, dim), requires_grad=True)
        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_transformer = Transformer(dim, depth, heads, dim_head, dim * scale_dim, dropout)

        self.interaction_token = nn.Parameter(torch.randn(1, 1, dim))
        self.interaction_embedding = nn.Parameter(torch.zeros(1, self.people + 1, dim))
        self.interaction_transformer = Transformer(dim, depth_int, heads, dim_head, dim * scale_dim, dropout)

        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, scale_dim * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(scale_dim * dim, num_classes)
        )

        self.num_channels = [dim // 2, dim]
        self.kernel_size = kernel_size
        input_size = 2

        layers = []
        num_levels = len(self.num_channels)
        for i in range(num_levels):
            dilation_size = (2 * i) + 1
            in_channels = input_size if i == 0 else self.num_channels[i - 1]
            out_channels = self.num_channels[i]
            padding = (kernel_size - 1) * dilation_size // 2
            layers += [
                nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation_size, padding=padding),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
                nn.Dropout(dropout)
            ]
        self.tcn1 = nn.Sequential(*layers)
        self.tcn2 = nn.Sequential(*layers)
        self.tcn3 = nn.Sequential(*layers)

        self.initialize_weights()

    def initialize_weights(self):
        temp_embed = repeat(
            torch.from_numpy(get_2d_sincos_pos_embed(self.temporal_embedding.shape[-1], int(self.time_sequence), cls_token=True)).float().unsqueeze(0),
            '() t d -> n t d', n=self.people
        )
        self.temporal_embedding.data.copy_(temp_embed.unsqueeze(0))

        int_embed = get_2d_sincos_pos_embed(self.interaction_embedding.shape[-1], self.people, cls_token=True)
        self.interaction_embedding.data.copy_(torch.from_numpy(int_embed).float().unsqueeze(0))

        torch.nn.init.normal_(self.interaction_token, std=.02)
        torch.nn.init.normal_(self.temporal_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, sp, t_pad=None):
        x = rearrange(x, 'b t n d -> b n t d')
        x = self.to_patch_embedding(x)
        b, n, t, _ = x.shape

        input_seq = rearrange(sp, 'b t d -> b d t')
        n = self.people

        x_pos1 = rearrange(self.tcn1(input_seq[:, :2, :]), 'b d t -> b t d').unsqueeze(1)
        x_pos2 = rearrange(self.tcn2(input_seq[:, 2:4, :]), 'b d t -> b t d').unsqueeze(1)
        x_shuttle = rearrange(self.tcn3(input_seq[:, -2:, :]), 'b d t -> b t d').unsqueeze(1)

        x = torch.cat((x, x_pos1, x_pos2, x_shuttle), dim=1)

        cls_temporal_tokens = repeat(self.temporal_token, '() t d -> b n t d', b=b, n=n)
        x = torch.cat((cls_temporal_tokens, x), dim=2)
        x += self.temporal_embedding[:, :, :(t + 1)]
        x = self.dropout(x)

        x = rearrange(x, 'b n t d -> (b n) t d')

        mask_t = torch.ones((b * n, t + 1), device=x.device).long()
        if t_pad is not None:
            t_pad = repeat(t_pad, '(t) -> (t d)', d=n).long().to(x.device)
            for j, index in enumerate(t_pad):
                mask_t[j, (index + 1):] = 0

        x = self.temporal_transformer(x, mask_t.unsqueeze(1).unsqueeze(1))
        x = rearrange(x[:, 0], '(b n) ... -> b n ...', b=b)

        cls_interaction_tokens = repeat(self.interaction_token, '() t d -> b t d', b=b)
        x = torch.cat((cls_interaction_tokens, x), dim=1)
        x += self.interaction_embedding[:, :(n + 1)]
        x = self.interaction_transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        return self.mlp_head(x)

    def predict(self, x, sp, pad=None):
        device = next(self.parameters()).device
        x = x.to(device)
        sp = sp.to(device)
        if pad is not None:
            pad = pad.to(device)
        pred = F.softmax(self.forward(x, sp, pad), dim=1).max(1)[1].cpu()
        return pred


# helpers
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemPoseII_TF(
        poses_numbers=94,
        time_steps=50,
        num_people=2,
        num_classes=13,
        dim=4,
        depth=3,
        depth_int=200,
        dim_head=128,
        emb_dropout=0.2
    )
    model.to(device)
    print(model)
