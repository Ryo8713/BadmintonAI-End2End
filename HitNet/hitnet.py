import torch
import torch.nn as nn
import torch.nn.functional as F

class HitNet(nn.Module):
    def __init__(self, input_dim, num_consec):
        super().__init__()
        self.num_consec = num_consec
        self.reshape_dim = input_dim // num_consec  # e.g. 936 // 36 = 26

        self.conv1 = nn.Conv1d(in_channels=self.reshape_dim, out_channels=128, kernel_size=3, padding=1)
        self.batchnorm = nn.BatchNorm1d(128)

        self.bigru = nn.GRU(
            input_size=128,
            hidden_size=256,
            num_layers=2,
            dropout=0.3,
            bidirectional=True,
            batch_first=True
        )

        self.query_layer = nn.Linear(512, 128)
        self.key_layer = nn.Linear(512, 128)
        self.value_layer = nn.Linear(512, 128)
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)

        self.pool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Linear(128, 256)
        self.dropout = nn.Dropout(0.3)
        self.output = nn.Linear(256, 1)

    def forward(self, x):
        # x shape: (batch_size, 936)
        x = x.view(x.size(0), self.num_consec, self.reshape_dim)  # (B, T, F)
        x = x.transpose(1, 2)  # (B, F, T) for Conv1D
        x = F.relu(self.conv1(x))
        x = self.batchnorm(x)
        x = x.transpose(1, 2)  # (B, T, F) for GRU

        gru_out, _ = self.bigru(x)  # (B, T, 512)
        query = self.query_layer(gru_out)
        key = self.key_layer(gru_out)
        value = self.value_layer(gru_out)

        attn_out, _ = self.attention(query, key, value)  # (B, T, 128)
        attn_out = attn_out.transpose(1, 2)  # (B, 128, T)
        pooled = self.pool(attn_out).squeeze(-1)  # (B, 128)

        x = F.relu(self.fc1(pooled))
        x = self.dropout(x)
        x = self.output(x)
        return x
