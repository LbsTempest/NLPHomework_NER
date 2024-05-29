import torch
import torch.nn as nn

from layers import PositionalEncoding, DecoderLayer


class Decoder(nn.Module):
    def __init__(
        self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len, output_dim, dropout=0.1
    ):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, input_ids, attention_mask):
        # Compute token and position embeddings
        token_embeddings = self.embedding(input_ids)
        position_embeddings = self.positional_encoding(token_embeddings)

        # Combine embeddings
        embeddings = token_embeddings + position_embeddings
        embeddings = self.dropout(embeddings)

        # Pass through decoder layers
        for layer in self.decoder_layers:
            embeddings = layer(embeddings, embeddings, attention_mask, attention_mask)

        logits = self.fc(embeddings)
        return logits


# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=512):
#         super(PositionalEncoding, self).__init__()
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(
#             torch.arange(0, d_model, 2).float()
#             * (-torch.log(torch.tensor(10000.0)) / d_model)
#         )
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer("pe", pe)

#     def forward(self, x):
#         return x + self.pe[: x.size(0), :]


# class MultiHeadAttention(nn.Module):
#     def __init__(self, d_model, num_heads):
#         super(MultiHeadAttention, self).__init__()
#         assert d_model % num_heads == 0
#         self.d_k = d_model // num_heads
#         self.num_heads = num_heads

#         self.linear_q = nn.Linear(d_model, d_model)
#         self.linear_k = nn.Linear(d_model, d_model)
#         self.linear_v = nn.Linear(d_model, d_model)
#         self.linear_out = nn.Linear(d_model, d_model)

#         self.attn_dropout = nn.Dropout(p=0.1)

#     def forward(self, query, key, value, mask=None):
#         batch_size = query.size(0)

#         query = (
#             self.linear_q(query)
#             .view(batch_size, -1, self.num_heads, self.d_k)
#             .transpose(1, 2)
#         )
#         key = (
#             self.linear_k(key)
#             .view(batch_size, -1, self.num_heads, self.d_k)
#             .transpose(1, 2)
#         )
#         value = (
#             self.linear_v(value)
#             .view(batch_size, -1, self.num_heads, self.d_k)
#             .transpose(1, 2)
#         )

#         scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
#             torch.tensor(self.d_k).float()
#         )
#         if mask is not None:
#             scores = scores.masked_fill(mask == 0, -1e9)
#         attn = torch.softmax(scores, dim=-1)
#         attn = self.attn_dropout(attn)
#         output = torch.matmul(attn, value)

#         output = (
#             output.transpose(1, 2)
#             .contiguous()
#             .view(batch_size, -1, self.num_heads * self.d_k)
#         )
#         output = self.linear_out(output)
#         return output


# class DecoderLayer(nn.Module):
#     def __init__(self, d_model, num_heads, d_ff, dropout):
#         super(DecoderLayer, self).__init__()
#         self.self_attn = MultiHeadAttention(d_model, num_heads)
#         self.ffn = nn.Sequential(
#             nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)
#         )
#         self.layernorm1 = nn.LayerNorm(d_model)
#         self.layernorm2 = nn.LayerNorm(d_model)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)

#     def forward(self, tgt, mask):
#         tgt2 = self.self_attn(tgt, tgt, tgt, mask)
#         tgt = tgt + self.dropout1(tgt2)
#         tgt = self.layernorm1(tgt)

#         tgt2 = self.ffn(tgt)
#         tgt = tgt + self.dropout2(tgt2)
#         tgt = self.layernorm2(tgt)
#         return tgt
