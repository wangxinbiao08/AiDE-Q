import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


def orthogonal_loss(weight_matrix):
    """
    Calculate the orthogonal loss: making the weight matrix close to unitary 
    """
    identity_matrix = torch.eye(weight_matrix.size(0)).to(weight_matrix.device)

    weight_transpose = weight_matrix.transpose(0, 1)
    loss = torch.norm(torch.matmul(weight_matrix, weight_transpose) - identity_matrix, 'fro')  # Frobenius norm
    return loss


class LSTM_Layer(nn.Module):
    def __init__(self, input_dim, emb_dim, dropout=0.3):
        super(LSTM_Layer, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=emb_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(emb_dim)
    def forward(self, x):
        output, _ = self.lstm(x)
        output = self.dropout(output)
        output = self.layer_norm(output)
        return output


class EmbeddingLayer(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbeddingLayer, self).__init__()
        # Define linear mapping layer, map qubit-dim  -> 128
        self.linear = nn.Linear(input_dim, emb_dim)

    def forward(self, x):
        # x 的 shape 为 (batch_size, seq_len, input_dim) => (32, 256, 50)
        x = self.linear(x)  # 将每个 50 维度映射到 128 维
        return x


class PositionEmbedding(nn.Module):
    def __init__(self, emb_dim, max_len):
        super(PositionEmbedding, self).__init__()
        # 仅创建位置嵌入
        self.position_embedding = self.create_position_embedding(max_len, emb_dim)
        self.param_J_projection = nn.Linear(2, emb_dim)

    @staticmethod
    def create_position_embedding(max_len, emb_dim):
        # 生成位置编码
        position_encoding = torch.zeros(max_len, emb_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (-math.log(10000.0) / emb_dim))
        position_encoding[:, 0::2] = torch.sin(position * div_term)
        position_encoding[:, 1::2] = torch.cos(position * div_term)
        position_encoding = position_encoding.unsqueeze(0)  # 添加批量维度
        return nn.Parameter(position_encoding, requires_grad=False)

    def forward(self, x, param_J):
        seq_len = x.size(1)
        position_emb = self.position_embedding[:, :seq_len, :]

        param_J_emb = self.param_J_projection(param_J)  # (batch, emb_dim)
        param_J_emb = param_J_emb.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, seq_len, emb_dim)

        return x + position_emb + param_J_emb


class Transformer_layer(nn.Module):
    def __init__(self, emb_dim, num_heads, dim_feedforward, num_layers, dropout=0.3):
        super(Transformer_layer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = x.permute(1, 0, 2)  # Transformer expects shape (seq_len, batch_size, emb_dim)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)
        return x


class LinearProjection(nn.Module):
    def __init__(self, emb_dim, seq_len, output_dim):
        super(LinearProjection, self).__init__()
        # 将 (seq_len * emb_dim) 映射到 output_dim
        self.linear = nn.Linear(seq_len * emb_dim, output_dim)

    def forward(self, x):
        batch_size, seq_len, emb_dim = x.shape
        # 将输入形状 (batch_size, seq_len, emb_dim) 展平为 (batch_size, seq_len * emb_dim)
        x = x.reshape(batch_size, -1)
        x = self.linear(x)  # 映射到 [batch_size, output_dim]
        return x


class Qtype(nn.Module):
    def __init__(self, n_qubits, num_measurements, shots, emb_dim, num_heads, num_layers, dim_feedforward):
        super(Qtype, self).__init__()
        self.LSTM_layer = LSTM_Layer(emb_dim, emb_dim)
        self.embedding_layer = EmbeddingLayer(n_qubits, emb_dim)
        self.position_embedding_layer = PositionEmbedding(emb_dim=emb_dim, max_len=shots)
        self.Transformer_layer = Transformer_layer(emb_dim, num_heads, dim_feedforward, num_layers)
        self.LinearProjection_layer = LinearProjection(emb_dim, shots, n_qubits)

    def forward(self, x, param_J):
        x = self.embedding_layer(x)
#        param_J = param_J.unsqueeze(1)
        x = self.position_embedding_layer(x, param_J)

        # 在 train 模式下添加噪声
        if self.training:  # 判断当前是否是训练模式
            noise = torch.randn_like(x) * 0.05  # 生成与输入相同大小的高斯噪声
            x = x + noise  # 将噪声加到输入数据中

        x = self.LSTM_layer(x)
        x = self.Transformer_layer(x)
        x = self.LinearProjection_layer(x)

        lstm_weight_ih = self.LSTM_layer.lstm.weight_ih_l0  # 输入到隐层的权重矩阵
        orthogonal_loss_lstm = orthogonal_loss(lstm_weight_ih)
        lstm_weight_hh = self.LSTM_layer.lstm.weight_hh_l0  # 隐层到隐层的权重矩阵
        orthogonal_loss_lstm_hh = orthogonal_loss(lstm_weight_hh)

        # 计算 transformer 权重的正交化损失
        transformer_weights = self.Transformer_layer.transformer_encoder.layers[
            0].self_attn.in_proj_weight  # self-attention 中的权重矩阵
        query_weight, key_weight, value_weight = torch.split(transformer_weights, transformer_weights.size(0) // 3, dim=0)
        orthogonal_loss_transformer = orthogonal_loss(query_weight) + orthogonal_loss(key_weight) + orthogonal_loss(value_weight)

        # 计算 linear 权重的正交化损失
        linear_weights = self.LinearProjection_layer.linear.weight
        orthogonal_loss_linear = orthogonal_loss(linear_weights)

        total_orthogonal_loss = orthogonal_loss_lstm + orthogonal_loss_lstm_hh + orthogonal_loss_transformer + orthogonal_loss_linear
        return x, total_orthogonal_loss





