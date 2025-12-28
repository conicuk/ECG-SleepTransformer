import torch.nn as nn
from utils.sp import SimPool

class CNNTransformerModel2D(nn.Module):
    def __init__(self, input_channels, num_classes, cnn_channels, transformer_dim, num_heads, num_layers, ff_dim, dropout=0.1, use_cnn=True, use_transformer=True):
        super().__init__()
        self.use_cnn = use_cnn
        self.use_transformer = use_transformer

        if self.use_cnn:
            self.mel_block1 = nn.Sequential(
                nn.Conv2d(input_channels, cnn_channels * 2, kernel_size=3, padding=1),
                nn.GELU(),
            )


            self.mel_block2 = nn.Sequential(
                nn.Conv2d(cnn_channels * 2, cnn_channels * 2, kernel_size=3, padding=1),
                nn.GELU(),
            )


            self.mel_block3 = nn.Sequential(
                nn.Conv2d(cnn_channels * 2, cnn_channels * 4, kernel_size=3, padding=1),
                nn.GELU(),
            )

        self.simpool = SimPool(cnn_channels * 4, gamma=None)

        if self.use_transformer:
            final_ch = cnn_channels * 4
            self.embedding = nn.Linear(final_ch, transformer_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=transformer_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        final_dim = transformer_dim if self.use_transformer else (cnn_channels*4)
        self.fc = nn.Linear(final_dim, num_classes)

    def forward(self, x):
        if self.use_cnn:
            x = self.mel_block1(x)
            x = self.mel_block2(x)
            x = self.mel_block3(x)

            x = self.simpool(x)

        else:
            x = x.squeeze(1)
            x = x.unsqueeze(2)
            x = self.simpool(x)

        if self.use_transformer:
            x = self.embedding(x).unsqueeze(1)
            x = self.transformer(x)
            x = x.squeeze(1)

        out = self.fc(x)
        return out

def Proposed(use_cnn=True, use_transformer=True):
    return CNNTransformerModel2D(
        input_channels=1,
        cnn_channels=16,
        num_classes=4,
        transformer_dim=128,
        num_heads=8,
        num_layers=2,
        ff_dim=256,
        dropout=0.1,
        use_cnn=use_cnn,
        use_transformer=use_transformer,
    )


