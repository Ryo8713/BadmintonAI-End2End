import torch
import torch.nn as nn
import torch.nn.functional as F

class HitNet(nn.Module):
    def __init__(self, input_dim, num_consec):
        super().__init__()
        self.num_consec = num_consec

        # Temporal CNN Block
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.batchnorm = nn.BatchNorm1d(128)

        # Bi-GRU
        self.bigru = nn.GRU(
            input_size=128,
            hidden_size=256,
            num_layers=2,
            dropout=0.5,
            bidirectional=True,
            batch_first=True
        )

        # Self-Attention
        self.query_layer = nn.Linear(512, 128)
        self.key_layer = nn.Linear(512, 128)
        self.value_layer = nn.Linear(512, 128)
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)

        # Pooling
        self.pool = nn.AdaptiveMaxPool1d(1)

        # Classifier
        self.fc1 = nn.Linear(256, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.output = nn.Linear(256, 1)

    def forward(self, x):
        # x: (B, T, F) = (B, 7, 70)
        x = x.transpose(1, 2)  # → (B, F, T)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))  # Second conv
        x = self.batchnorm(x)
        x = x.transpose(1, 2)  # → (B, T, F) = (B, 7, 128)

        gru_out, _ = self.bigru(x)  # (B, T, 512)

        # Attention
        query = self.query_layer(gru_out)
        key = self.key_layer(gru_out)
        value = self.value_layer(gru_out)
        attn_out, _ = self.attention(query, key, value)  # (B, T, 128)

        # Combine GRU output with attention (residual)
        fused = attn_out + query  # (B, T, 128)
        fused = fused.transpose(1, 2)  # → (B, 128, T)

        # Use both max and avg pooling
        max_pooled = self.pool(fused).squeeze(-1)
        avg_pooled = fused.mean(dim=2)
        pooled = torch.cat([max_pooled, avg_pooled], dim=1)  # (B, 256)

        # Classifier
        x = F.relu(self.fc1(pooled))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.output(x)
        return x

