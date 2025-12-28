import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

class DCRModel(nn.Module):
    def __init__(self, input_channel, num_classes):
        super(DCRModel, self).__init__()

        self.batchnorm = nn.BatchNorm1d(1)

        self.conv1_1 = nn.Conv1d(in_channels=input_channel, out_channels=60, kernel_size=50, stride=1)
        self.pool1_1 = nn.MaxPool1d(kernel_size=2, stride=1)
        self.dropout1_1 = nn.Dropout(0.15)

        self.conv2_1 = nn.Conv1d(in_channels=60, out_channels=30, kernel_size=30, stride=1)
        self.pool2_1 = nn.MaxPool1d(kernel_size=2, stride=1)
        self.dropout2_1 = nn.Dropout(0.15)

        self.conv3_1 = nn.Conv1d(in_channels=30, out_channels=10, kernel_size=20, stride=1)
        self.pool3_1 = nn.MaxPool1d(kernel_size=2, stride=1)
        self.dropout3_1 = nn.Dropout(0.15)

        self.gru1 = nn.GRU(input_size=10, hidden_size=20, num_layers=1, batch_first=True)
        self.dropout_gru1 = nn.Dropout(0.15)

        self.gru2 = nn.GRU(input_size=20, hidden_size=10, num_layers=1, batch_first=True)
        self.dropout_gru2 = nn.Dropout(0.15)

        self.fc = nn.Linear(10, num_classes)


    def forward(self, x):
        x = self.batchnorm(x)

        x = self.conv1_1(x)
        x = self.pool1_1(x)
        x = F.relu(x)
        x = self.dropout1_1(x)

        x = self.conv2_1(x)
        x = self.pool2_1(x)
        x = F.relu(x)
        x = self.dropout2_1(x)

        x = self.conv3_1(x)
        x = self.pool3_1(x)
        x = F.relu(x)
        x = self.dropout3_1(x)

        x = x.permute(0, 2, 1)

        x, _ = self.gru1(x)
        x = self.dropout_gru1(x)
        x, _ = self.gru2(x)
        x = self.dropout_gru2(x)

        x = x[:, -1, :]

        output = self.fc(x)  # 4-class
        return output

class IHR_EDR_Model(nn.Module):
    def __init__(self, input_dim=1, num_classes=4):
        super(IHR_EDR_Model, self).__init__()
        conv_layers = []
        in_ch = input_dim
        for _ in range(6):
            conv_layers += [
                nn.Conv1d(in_ch, 64, kernel_size=6, stride=1, padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2)
            ]
            in_ch = 64

        self.ihr_blocks = nn.Sequential(*conv_layers)
        self.edr_blocks = nn.Sequential(*conv_layers)

        self.ihr_gru = nn.GRU(input_size=64, hidden_size=256, batch_first=True, bidirectional=True)
        self.ihr_fc = nn.Linear(512, 128)

        self.edr_gru = nn.GRU(input_size=64, hidden_size=256, batch_first=True, bidirectional=True)
        self.edr_fc = nn.Linear(512,128)

        self.classifier= nn.Linear(256, num_classes)

    def forward(self, ihr_data, edr_data):
        x_ihr = None
        x_edr = None

        x_ihr = self.ihr_blocks(ihr_data)

        x_ihr = x_ihr.permute(0, 2, 1)
        x_ihr_gru, _ = self.ihr_gru(x_ihr)
        x_ihr = x_ihr_gru[:, -1, :]

        x_ihr = self.ihr_fc(x_ihr)

        x_edr = self.edr_blocks(edr_data)

        x_edr = x_edr.permute(0, 2, 1)
        x_edr_gru, _ = self.edr_gru(x_edr)
        x_edr = x_edr_gru[:, -1, :]

        x_edr = self.edr_fc(x_edr)

        x_ihr_edr = torch.cat([x_ihr, x_edr], dim=1)

        x = self.classifier(x_ihr_edr)
        return x

class ConvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.0):
        super(ConvBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=1)
        self.act1 = nn.LeakyReLU(negative_slope=0.15)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=1)
        self.act2 = nn.LeakyReLU(negative_slope=0.15)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(dropout)

        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        res = self.residual(self.pool(x))
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x + res


class DilatedConvBlock(nn.Module):
    def __init__(self, channels, dropout=0.2):
        super(DilatedConvBlock, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size=7, padding='same', dilation=d)
            for d in [2, 4, 8, 16, 32]
        ])
        self.acts = nn.ModuleList([
            nn.LeakyReLU(negative_slope=0.15) for _ in range(5)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        res = x
        for conv, act in zip(self.convs, self.acts):
            x = conv(x)
            x = act(x)
        x = self.dropout(x)
        return x + res


class SleepStageCNN(nn.Module):
    def __init__(self, input_dim=1, num_classes=4):
        super(SleepStageCNN, self).__init__()

        self.input_conv = nn.Conv1d(input_dim, 8, kernel_size=1)

        self.block1 = ConvBlock1D(8, 16, dropout=0.1)
        self.block2 = ConvBlock1D(16, 32, dropout=0.1)
        self.block3 = ConvBlock1D(32, 64, dropout=0.1)

        self.flatten_fc = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(64 * 32, 128),
        )

        self.dilated1 = DilatedConvBlock(128, dropout=0.2)
        self.dilated2 = DilatedConvBlock(128, dropout=0.2)

        self.output_conv = nn.Conv1d(128, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.input_conv(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.flatten_fc(x)
        x = x.unsqueeze(0)
        x = x.permute(0, 2, 1)

        x = self.dilated1(x)
        x = self.dilated2(x)

        out = self.output_conv(x)
        out = out.permute(2, 0, 1)
        return out.squeeze(1)


class Config:
    def __init__(self, epoch_seq_len=20, nclass=5, ndim=128, nchannel=1, frame_seq_len=29, frm_num_blocks=2, seq_num_blocks=2):
        self.ndim = ndim
        self.frame_seq_len = frame_seq_len
        self.epoch_seq_len = epoch_seq_len
        self.nchannel = nchannel
        self.nclass = nclass

        self.frame_attention_size = 64

        self.frm_d_model = self.ndim * self.nchannel
        self.frm_d_ff = 1024
        self.frm_num_blocks = frm_num_blocks
        self.frm_num_heads = 8
        self.frm_maxlen = frame_seq_len
        self.frm_fc_dropout = 0.1
        self.frm_attention_dropout = 0.1

        self.seq_d_model = self.ndim * self.nchannel
        self.seq_d_ff = 1024
        self.seq_num_blocks = seq_num_blocks
        self.seq_num_heads = 8
        self.seq_maxlen = epoch_seq_len
        self.seq_fc_dropout = 0.1
        self.seq_attention_dropout = 0.1

        self.fc_hidden_size = 1024
        self.fc_dropout = 0.1

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-8): #
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return self.a_2 * (x - mean) / torch.sqrt(var + self.eps) + self.b_2

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)].requires_grad_(False)

def ScaledDotProductAttention(query, key, value, dropout=None, training=True):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None and training:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value)

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout_rate=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model, bias=True), 4)
        self.attn_dropout = nn.Dropout(p=dropout_rate)
        self.layer_norm = LayerNorm(d_model)

    def forward(self, query, key, value, training=True):
        N = query.size(0)
        d_model = query.size(-1)
        residual = query

        query, key, value = [l(x).view(N, -1, self.h, self.d_k).transpose(1, 2).reshape(-1, x.size(1), self.d_k)
                             for l, x in zip(self.linears[:3], (query, key, value))]

        x = ScaledDotProductAttention(query, key, value, dropout=self.attn_dropout, training=training)

        x = x.view(N, -1, self.h, self.d_k).transpose(1, 2).reshape(N, -1, d_model)
        x = self.linears[-1](x)

        x = x + residual

        return self.layer_norm(x)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout_rate=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = LayerNorm(d_model)

    def forward(self, x, training=True):
        residual = x
        x = F.relu(self.w_1(x))
        x = self.w_2(x)

        x = x + residual

        x = self.layer_norm(x)

        x = self.dropout(x) if training else x

        return x

class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, fc_dropout_rate, attention_dropout_rate):
        super(EncoderBlock, self).__init__()
        self.self_attn = MultiHeadedAttention(num_heads, d_model, attention_dropout_rate)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, fc_dropout_rate)

    def forward(self, x, training=True):
        x = self.self_attn(x, x, x, training=training)
        return self.feed_forward(x, training=training)

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, d_ff, num_blocks, num_heads, maxlen, fc_dropout_rate, attention_dropout_rate):
        super(TransformerEncoder, self).__init__()
        block = EncoderBlock(d_model, num_heads, d_ff, fc_dropout_rate, attention_dropout_rate)
        self.blocks = clones(block, num_blocks)
        self.pos_encoding = PositionalEncoding(d_model, max_len=maxlen)
        self.d_model = d_model

    def forward(self, x, training=True):
        x = x * math.sqrt(self.d_model)
        x = x + self.pos_encoding(x)

        for block in self.blocks:
            x = block(x, training=training)

        return x

class Attention(nn.Module):
    def __init__(self, hidden_size, attention_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attention_size = attention_size

        self.W_omega = nn.Parameter(torch.Tensor(hidden_size, attention_size))
        self.b_omega = nn.Parameter(torch.Tensor(attention_size))
        self.u_omega = nn.Parameter(torch.Tensor(attention_size))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 0.1
        nn.init.normal_(self.W_omega, std=stdv)
        nn.init.normal_(self.b_omega, std=stdv)
        nn.init.normal_(self.u_omega, std=stdv)

    def forward(self, inputs):
        v = torch.tanh(torch.matmul(inputs.reshape(-1, self.hidden_size), self.W_omega) + self.b_omega)

        vu = torch.matmul(v, self.u_omega.unsqueeze(-1))

        exps = torch.exp(vu.view(inputs.size(0), inputs.size(1)))

        alphas = exps / torch.sum(exps, dim=1, keepdim=True)

        output = torch.sum(inputs * alphas.unsqueeze(-1), dim=1)

        return output, alphas


class SleepTransformer(nn.Module):
    def __init__(self, config):
        super(SleepTransformer, self).__init__()
        self.config = config

        self.frame_transformer = TransformerEncoder(
            d_model=config.frm_d_model, d_ff=config.frm_d_ff, num_blocks=config.frm_num_blocks,
            num_heads=config.frm_num_heads, maxlen=config.frm_maxlen,
            fc_dropout_rate=config.frm_fc_dropout, attention_dropout_rate=config.frm_attention_dropout
        )

        self.frame_attention = Attention(
            hidden_size=config.frm_d_model,
            attention_size=config.frame_attention_size
        )

        self.seq_transformer = TransformerEncoder(
            d_model=config.seq_d_model, d_ff=config.seq_d_ff, num_blocks=config.seq_num_blocks,
            num_heads=config.seq_num_heads, maxlen=config.seq_maxlen,
            fc_dropout_rate=config.seq_fc_dropout, attention_dropout_rate=config.seq_attention_dropout
        )

        self.fc1 = nn.Linear(config.seq_d_model, config.fc_hidden_size)
        self.dropout1 = nn.Dropout(config.fc_dropout)
        self.fc2 = nn.Linear(config.fc_hidden_size, config.fc_hidden_size)
        self.dropout2 = nn.Dropout(config.fc_dropout)
        self.output_linear = nn.Linear(config.fc_hidden_size, config.nclass)

    def forward(self, x, istraining=True):
        N, T_e, T_f, D_d, D_c = x.shape

        frm_trans_X = x.reshape(-1, T_f, D_d * D_c)

        frm_trans_out = self.frame_transformer(frm_trans_X, training=istraining)
        attention_out, _ = self.frame_attention(frm_trans_out)
        seq_trans_X = attention_out.reshape(N, T_e, self.config.frm_d_model)

        seq_trans_out = self.seq_transformer(seq_trans_X, training=istraining)
        fc_in = seq_trans_out.reshape(-1, self.config.seq_d_model)

        fc1 = F.relu(self.fc1(fc_in))
        fc1 = self.dropout1(fc1) if istraining else fc1
        fc2 = F.relu(self.fc2(fc1))
        fc2 = self.dropout2(fc2) if istraining else fc2
        score = self.output_linear(fc2)

        scores = score.reshape(N, T_e, self.config.nclass)
        predictions = torch.argmax(scores, dim=-1)

        return scores, predictions

def Baseline1(input_channel=1, num_classes=4):  #Baseline 1 [Urtnasan et al. [46]]
    return DCRModel(input_channel=input_channel, num_classes=num_classes)

def Baseline2(input_dim=1, num_classes=4): #Baseline 2 [Sharan et al. [48]]
    return IHR_EDR_Model(input_dim=input_dim, num_classes=num_classes)

# 사용자용 생성 함수
def Baseline3(input_dim=1, num_classes=4): #Baseline 3 [Sridhar et al. [49]]
    return SleepStageCNN(input_dim=input_dim, num_classes= num_classes)

def Baseline4(epoch_seq_len=20, nclass=4, ndim=128, nchannel=1, frame_seq_len=29, frm_num_blocks=2, seq_num_blocks=2): #Basline 4 [Phan et al. [47]]
    config = Config(
        epoch_seq_len=epoch_seq_len, nclass=nclass, ndim=ndim, nchannel=nchannel,
        frame_seq_len=frame_seq_len, frm_num_blocks=frm_num_blocks, seq_num_blocks=seq_num_blocks
    )
    return SleepTransformer(config)