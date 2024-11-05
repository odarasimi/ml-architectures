"""

The transformer architecture.

Link to paper
    https://arxiv.org/abs/1706.03762
Reference(s)
    https://github.com/hkproj/pytorch-transformer/blob/main/model.py
Note!
    For the residual connection, this implementation applies layer normalization 
    AFTER the 'multi-head attention' and 'feed-forward network' layers, not BEFORE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, List, Dict, Optional, Tuple, Any
from einops import rearrange, reduce, repeat

class InputEmbeddings(nn.Module):
    def __init(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor):
        return self.embedding(x) * np.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super.__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.seq_len = seq_len

        # create a positional encoding matrix
        # pos_encoding shape: each token of sequence in a row and dimension of tokens determines columns = seq_len X d_model
        pe = torch.randn(seq_len, d_model)
        pos = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # einops: pos = rearrange(torch.arange(0, 20, dtype=torch.float), "seq_len -> seq_len ()")

        # computing the denominator: 10000 is a large number, so we use exp for numerical stability - avoid overflows, better precision,
        # compute efficient and all that
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        #populate the positional encoding matrix
        # apply the sin function for each even indexed column in the token row
        pe[:, 0::2] =  torch.sin(pos * div_term)
        # apply the cosine function for each odd indexed column in the token row
        pe[:, 1::2] = torch.cos(pos * div_term)
        # we will have a batch of sentences, so add batch dimension (1, seq_len, d_model)
        pe = pe.unsqueeze(0) # einops: pe = rearrange(pe, "seq_len d_model -> () seq_len d_model")

        # save the tensor for later use, remember pe is fixed
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor):
        # add positional encoding to input embedding
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # a forward autograd graph will not be created for this tensor
        x = self.dropout(x)
        return (x)

class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps # we add this to the variance, so denominator is never 0
        self.affine = affine
        if self.affine:
            # affine tranformation/reparametrization trick to add the possibility of some variability in var and mean
            self.alpha = nn.Parameter(torch.ones(1))
            self.bias = nn.Parameter(torch.zeros(1))
        else:
            self.alpha = None
            self.bias = None

    def forward(self, x: torch.Tensor):
        mean = x.mean(dim = -1, keepdim = True) # shape: (batch_size, seq_length, 1)
        std = x.std(dim = -1, keepdim = True) # shape: (batch_size, seq_length, 1)
        norm = (x - mean) / (std + self.eps)
        if self.affine:
            return (self.alpha * (norm) + self.bias)
        else:
            return norm

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.linear_layer1 = nn.Linear(d_model, hidden_dim)
        self.linear_layer2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        x = self.dropout(F.gelu(self.linear_layer1(x)))
        x = self.linear_layer2(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float = 0.1 ):
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        assert d_model % heads == 0, "d_model must be divisible by heads"

        self.d_k = d_model // heads

        self.w_q = nn.Linear(d_model, d_model)
        torch.nn.init.xavier_uniform(self.w_q.weight)

        self.w_k = nn.Linear(d_model, d_model)
        torch.nn.init.xavier_uniform(self.w_k.weight)

        self.w_v = nn.Linear(d_model, d_model)
        torch.nn.init.xavier_uniform(self.w_v.weight)

        self.w_o = nn.Linear(d_model, d_model)
        torch.nn.init.xavier_uniform(self.w_o.weight)

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        score = torch.matmul(query, key.permute(0, 1, 3, 2)) / np.sqrt(d_k)

        if mask:
            score = score.masked_fill(mask == 0, -1e9)
            score = F.softmax(score, dim = -1)
            if dropout is not None:
                score = dropout(score)
            return torch.matmul(score, value)
        else:
            return torch.matmul(F.softmax(score, dim = -1), value)


    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask:torch.Tensor):
        batch_size = q.shape[0]

        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # (Batch, seq_len, d_model) --> (Batch, seq_len, heads, d_k) --> (Batch, heads, seq_len, d_k): heads are in the channel dimension
        query = query.view(query.shape[0], query.shape[1], self.heads, self.d_k).permute(0, 2, 1, 3)
        # einops makes it more readable: query = rearrange(query, "b s (h d) -> b h s d", h=self.heads)
        key = key.view(key.shape[0], key.shape[1], self.heads, self.d_k).permute(0, 2, 1, 3)
        # einops: key = rearrange(key, "b s (h d) -> b h s d", h=self.heads)
        value = value.view(value.shape[0], value.shape[1], self.heads, self.d_k).permute(0, 2, 1, 3)
        # einops: value = rearrange(value, "b s (h d) -> b h s d", h=self.heads)

        x = MultiHeadAttention.attention(query, key, value, mask, self.dropout)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h*self.d_k)
        # einops: x = rearrange(x, "b h s d -> b s (h d)")

        return self.w_o(x)

class ResidualConnection(nn.Module):
    def __init__(self, dropout:float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x:torch.Tensor, sublayer: torch.Tensor):
        return self.norm(self.dropout(x + sublayer(x)))

class EncoderBlock(nn.Module):
    def __init__(self,self_attention_layer: MultiHeadAttention, feed_forward_layer: FeedForwardNetwork, dropout: float):
        super().__init__()
        self.self_attention_layer = self_attention_layer
        self.feed_forward_layer = feed_forward_layer
        self.residual_connection1 = ResidualConnection(dropout)
        self.residual_connection2 = ResidualConnection(dropout)

    def forward(self, x:torch.Tensor, padding_mask:torch.Tensor):
        # x goes into the first self attention block to create the q, k, v
        x = self.residual_connection1(x, lambda x: self.self_attention_layer(x, x, x, padding_mask)) # passing a lambda function that takes in arg 'x'
        x = self.residual_attention2(x, self.feed_forward_layer)
        return x

class Encoder(nn.Module):
    def __init__(self, encoder_layer: nn.ModuleList):
        super().__init__()
        self.layers = encoder_layer
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, self_attention_layer: MultiHeadAttention, cross_attention_layer: MultiHeadAttention, feed_forward_layer: FeedForwardNetwork, dropout: float):
        super().__init__()
        self.self_attention_layer = self_attention_layer
        self.cross_attention_layer = cross_attention_layer
        self.feed_forward_layer = feed_forward_layer
        self.residual_connection1 = ResidualConnection(dropout)
        self.residual_connection2 = ResidualConnection(dropout)
        self.residual_connection3 = ResidualConnection(dropout)

    def forward(self, x:torch.Tensor, encoder_output: torch.Tensor, padding_mask: torch.Tensor, causal_mask: torch.Tensor):
        x = self.residual_connection1(x, lambda x: self.self_attention_layer(x, x, x, padding_mask))
        x = self.residual_connection2(x, lambda x: self.cross_attention_layer(x, encoder_output, encoder_output, causal_mask))
        x = self.residual_connection3(x, self.feed_forward_layer)
        return x

class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers

    def forward(self, x:torch.Tensor, encoder_output: torch.Tensor, padding_mask: torch.Tensor, causal_mask: torch.Tensor):
        for layer in self.layers:
            x = layer(x, encoder_output, padding_mask, causal_mask)
        return x

class OutputProjection(nn.Module): # Project embeddings back into the vocabulary
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.projection = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor):
        # transform each d_model vector into a vocab_size vector and produce log-probabilities over the vocabulary for each position in seq_len
        return torch.log_softmax(self.projection(x), dim = -1) # batch, seq_len, d_model --> batch, seq_len, vocab_size

class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: OutputProjection):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src: torch.Tensor, padding_mask: torch.Tensor):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, padding_mask)

    def decode(self, encoder_output: torch.Tensor, padding_mask: torch.Tensor, tgt: torch.Tensor, causal_mask: torch.Tensor):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, padding_mask, causal_mask)

    def project(self, x:torch.Tensor):
        return self.projection_layer(x)

def transform(src_size: int, tgt_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, num_enc_blocks: int = 6, heads: int = 8, dropout: float = 0.1, hidden_dim: int = 2048):
    src_embed = InputEmbeddings(d_model, src_size)
    tgt_embed = InputEmbeddings(d_model, tgt_size)

    src_pos_embed = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos_embed = PositionalEncoding(d_model, tgt_seq_len, dropout)

    encoder_blocks = [EncoderBlock(MultiHeadAttention(d_model, heads, dropout), FeedForwardNetwork(d_model, hidden_dim, dropout), dropout) for _ in range(num_enc_blocks)]
    decoder_blocks = [DecoderBlock(MultiHeadAttention(d_model, heads, dropout), MultiHeadAttention(d_model, heads, dropout), FeedForwardNetwork(d_model, hidden_dim, dropout), dropout) for _ in range(num_enc_blocks)]

    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    projection_layer = OutputProjection(d_model, tgt_size)
    return Transformer(encoder, decoder, src_embed, tgt_embed, src_pos_embed, tgt_pos_embed, projection_layer)