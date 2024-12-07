import math
from queue import PriorityQueue

import numpy as np
import torch
import torch.nn.functional as F
from sparsemax import Sparsemax
from torch import nn
from torch.nn.utils.rnn import pad_sequence


def transpose_qkv(X, num_heads):
    """
    Transposition for parallel computation of multiple attention heads.
    """
    if X.dim() == 4:
        b_, s_, t_, d_ = X.size()
        # Reshape to split the embedding dimension for each head
        X = X.reshape(b_, s_, t_, num_heads, -1)
        X = X.permute(1, 0, 3, 2, 4)
        # Flatten dimensions for efficient attention computation
        return X.reshape(s_, b_ * num_heads, t_, -1)
    else:
        # Reshape and split into heads
        X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
        X = X.permute(0, 2, 1, 3)
        # Flatten for further processing
        return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    """
    Reverse the operation of `transpose_qkv`.
    """
    # Reshape to separate heads
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    # Flatten heads into the embedding dimension
    return X.reshape(X.shape[0], X.shape[1], -1)


def masked_softmax(X, valid_lens):
    """
    Perform softmax operation by masking elements on the last axis.
    """
    if valid_lens is None: # If no valid lengths provided, perform regular softmax
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        # Reshape back and apply softmax
        return nn.functional.softmax(X.reshape(shape), dim=-1)


def masked_sparsemax(X, valid_lens):
    """
    Perform Sparsemax operation by masking elements on the last axis.
    """
    sparsemax = Sparsemax(dim=-1)
    if X.dim() == 3:
        shape = X.shape
        valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        # Apply sequence masking to set invalid positions to a large negative value
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        # Reshape back and apply Sparsemax
        return sparsemax(X.reshape(shape))
    elif X.dim() == 4:
        shape = X.shape
        valid_lens = torch.repeat_interleave(valid_lens.transpose(0, 1).reshape(-1), shape[2])
        # Apply sequence masking to set invalid positions to a large negative value
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return sparsemax(X.reshape(shape))
    else:
        assert 1 == 2 # Trigger an error if unsupported


def sequence_mask(X, valid_len, value=0):
    """
    Mask irrelevant entries in sequences.
    """
    if X.dim() == 4:
        maxlen = X.size(2)
        # Create a mask where positions less than valid_len are True
        mask = torch.arange((maxlen), dtype=torch.float32,
                            device=X.device)[None, :] < valid_len[:, :, None]
        X[~mask] = value
    else:
        maxlen = X.size(1)
        # Create a mask for valid lengths
        mask = torch.arange((maxlen), dtype=torch.float32,
                            device=X.device)[None, :] < valid_len[:, None]
        X[~mask] = value
    return X


class PositionWiseFFN(nn.Module):
    """
    Implements a Position-wise Feed-Forward Network (FFN).
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFFN, self).__init__()
        self.W_1 = nn.Linear(d_model, d_ff)
        self.W_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Apply the first linear transformation, followed by ReLU, dropout, and the second transformation
        return self.W_2(self.dropout(F.relu(self.W_1(x))))


class AddNorm(nn.Module):
    """
    Implements the Add & Norm layer used in Transformers.
    """
    def __init__(self, embedding_dim, dropout=0.1):
        super(AddNorm, self).__init__()
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y):
        # Apply residual connection (add x and y), dropout, and layer normalization
        return self.norm(x + self.dropout(y))


class PositionalEncoding(nn.Module):
    """
    Implements sinusoidal positional encoding for Transformers.
    """
    def __init__(self, d_model, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout) # Dropout for regularization
        self.P = torch.zeros((1, max_len, d_model)) # Positional encoding matrix
        # Compute position values scaled by sine and cosine
        x = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / \
            torch.pow(10000, torch.arange(0, d_model, 2, dtype=torch.float32) / d_model)
        self.P[:, :, 0::2] = torch.sin(x)
        self.P[:, :, 1::2] = torch.cos(x)

    def forward(self, x):
        # Add positional encoding to the input and apply dropout
        x = x + self.P[:, :x.shape[1], :].to(x.device)
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """
    Implements learnable positional encoding for Transformers.
    """
    def __init__(self, d_model, dropout, max_len=100):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x, pos):
        # Add positional embeddings to the input and apply dropout
        x = x + self.pos_embedding(pos)
        return self.dropout(x)


class MultiHeadAttentionWithRPR(nn.Module):
    """
    Multi-head attention with Relative Position Representations (RPR).
    """

    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, clipping_distance,
                 dropout, bias=False, **kwargs):
        super(MultiHeadAttentionWithRPR, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention_rpr = DotProductAttentionWithRPR(dropout)
        # Linear layers for projecting queries, keys, and values
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)
        # Embedding layers for relative position representations
        self.relative_pos_v = nn.Embedding(2 * clipping_distance + 1, num_hiddens // num_heads)
        self.relative_pos_k = nn.Embedding(2 * clipping_distance + 1, num_hiddens // num_heads)
        self.clipping_distance = clipping_distance

    def forward(self, queries, keys, values, valid_lens):
        # Project queries, keys, and values
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        # Calculate relative position distances
        range_queries = torch.arange(queries.size(1), device=queries.device)
        range_keys = torch.arange(keys.size(1), device=keys.device)
        distance_mat = range_keys[None, :] - range_queries[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.clipping_distance, self.clipping_distance) + \
                               self.clipping_distance
        # Get positional embeddings for keys and values
        pos_k = self.relative_pos_k(distance_mat_clipped)
        pos_v = self.relative_pos_v(distance_mat_clipped)

        # Repeat valid_lens for multiple heads if provided
        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        # Apply scaled dot-product attention with RPR
        output = self.attention_rpr(queries, keys, values, pos_k, pos_v, valid_lens)

        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


class DotProductAttentionWithRPR(nn.Module):
    """
    Scaled dot-product attention with Relative Position Representations(RPR).
    """

    def __init__(self, dropout, **kwargs):
        super(DotProductAttentionWithRPR, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, pos_k, pos_v, valid_lens=None):
        d = queries.shape[-1]
        # Compute attention scores from queries and keys
        scores = torch.bmm(queries, keys.transpose(1, 2))
        scores_pos = torch.bmm(queries.transpose(0, 1), pos_k.transpose(1, 2)).transpose(0, 1)
        scores = (scores + scores_pos) / math.sqrt(d)
        # Apply masked softmax to filter invalid positions
        self.attention_weights = masked_softmax(scores, valid_lens)
        # Compute weighted sum of values
        output = torch.bmm(self.dropout(self.attention_weights), values)
        output_pos = torch.bmm(self.dropout(self.attention_weights.transpose(0, 1)), pos_v).transpose(0, 1)
        return output + output_pos


class MultiHeadSelectiveAttention(nn.Module):
    """
    Selective attention using multi-head attention mechanism.
    """
    def __init__(self, query_size, key_size, value_size, num_hiddens, num_heads, stat_k, token_k, bias=False, **kwargs):
        super(MultiHeadSelectiveAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        # Linear layers for projecting queries, keys, and values
        self.W_q_stat = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_q_token = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k_stat = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_k_token = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)
        self.stat_k = stat_k
        self.token_k = token_k

    def forward(self, queries, stat_keys, token_keys, values, stat_valid_lens):
        # Project queries and keys
        query_stat = transpose_qkv(self.W_q_stat(queries), self.num_heads)
        query_token = transpose_qkv(self.W_q_token(queries), self.num_heads)
        key_stat = transpose_qkv(self.W_k_stat(stat_keys), self.num_heads)
        key_token = transpose_qkv(self.W_k_token(token_keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        # Compute static attention scores
        b_h_, query_num, d_ = query_stat.size()
        stat_num = key_stat.size(1)
        token_num = key_token.size(1)
        stat_scores = torch.bmm(query_stat, key_stat.transpose(1, 2)) / math.sqrt(d_)

        # Mask static scores for invalid positions
        if stat_valid_lens.dim() == 1:
            stat_valid_lens = torch.repeat_interleave(stat_valid_lens, self.num_heads * query_num)
            stat_scores = sequence_mask(stat_scores.reshape(-1, stat_num), stat_valid_lens, value=-1e6)
            stat_scores = stat_scores.reshape(b_h_, query_num, stat_num)
        else:
            assert 1 == 2

        # Select top-k static attention scores
        stat_k_weights, stat_k_indices = torch.topk(stat_scores, k=min(stat_num - 1, self.stat_k), dim=-1)
        stat_weights = torch.full_like(stat_scores, -1e6)
        stat_weights = stat_weights.scatter(-1, stat_k_indices, stat_k_weights)
        stat_weights = torch.softmax(stat_weights, dim=-1)

        # Compute token-level attention scores
        query_token = torch.repeat_interleave(query_token, stat_num, dim=0)
        key_token = key_token.reshape(-1, stat_num, self.num_heads, token_num, d_).transpose(1, 2).\
            reshape(b_h_ * stat_num, token_num, d_)
        token_scores = torch.bmm(query_token, key_token.transpose(1, 2)) / math.sqrt(d_)
        token_k_weights, token_k_indices = torch.topk(token_scores, k=min(token_num - 1, self.token_k), dim=-1)
        token_weights = torch.full_like(token_scores, -1e6)
        token_weights[:, :, -min(token_num - 1, self.token_k):] = token_scores[:, :, -min(token_num - 1, self.token_k):]
        token_weights = torch.softmax(token_weights, dim=-1)

        # Combine static and token weights
        token_weights = token_weights.reshape(b_h_, stat_num, query_num, token_num).transpose(1, 2)
        combine_weights = torch.mul(stat_weights.unsqueeze(-1), token_weights).reshape(b_h_, query_num, stat_num * token_num)
        values = values.reshape(-1, stat_num, self.num_heads, token_num, d_).transpose(1, 2).reshape(b_h_, stat_num * token_num, d_)

        # Compute final output
        output_concat = transpose_output(torch.bmm(combine_weights, values), self.num_heads)
        return self.W_o(output_concat)


class SelectiveAttention(nn.Module):
    """
    Selective Attention mechanism with hierarchical static and token-level attention.
    """
    def __init__(self, query_size, key_size, value_size, num_hiddens, stat_k, token_k, bias=False, **kwargs):
        super(SelectiveAttention, self).__init__(**kwargs)
        # Linear layers for projecting queries, keys, and values
        self.W_q_stat = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_q_token = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k_stat = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_k_token = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)
        self.stat_k = stat_k
        self.token_k = token_k

    def forward(self, queries, stat_keys, token_keys, values, stat_valid_lens):
        # Project queries and keys for static and token levels
        query_stat = self.W_q_stat(queries)
        query_token = self.W_q_token(queries)
        key_stat = self.W_k_stat(stat_keys)
        key_token = self.W_k_token(token_keys)
        values = self.W_v(values)
        b_, query_num, d_ = query_stat.size()
        stat_num = key_stat.size(1)
        token_num = key_token.size(1)

        # Compute static attention scores
        stat_scores = torch.bmm(query_stat, key_stat.transpose(1, 2)) / math.sqrt(d_)
        if stat_valid_lens.dim() == 1:
            # Mask invalid positions in the static attention scores
            stat_valid_lens = torch.repeat_interleave(stat_valid_lens, query_num)
            stat_scores = sequence_mask(stat_scores.reshape(-1, stat_num), stat_valid_lens, value=-1e6)
            stat_scores = stat_scores.reshape(b_, query_num, stat_num)
        else:
            assert 1 == 2
        
        # Select top-k static attention scores and normalize
        stat_k_weights, stat_k_indices = torch.topk(stat_scores, k=min(stat_num - 1, self.stat_k), dim=-1)
        stat_weights = torch.full_like(stat_scores, -1e6)
        stat_weights = stat_weights.scatter(-1, stat_k_indices, stat_k_weights)
        stat_weights = torch.softmax(stat_weights, dim=-1)

        # Repeat queries for token-level computation
        query_token = torch.repeat_interleave(query_token, stat_num, dim=0)
        # Compute token-level attention scores
        token_scores = torch.bmm(query_token, key_token.transpose(1, 2)) / math.sqrt(d_)
        token_k_weights, token_k_indices = torch.topk(token_scores, k=min(token_num - 1, self.token_k), dim=-1)
        token_weights = torch.full_like(token_scores, -1e6)
        token_weights = token_weights.scatter(-1, token_k_indices, token_k_weights)
        token_weights = torch.softmax(token_weights, dim=-1)
        token_weights = token_weights.reshape(b_, stat_num, query_num, token_num).transpose(1, 2)

        # Combine static and token-level attention weights
        combine_weights = torch.mul(stat_weights.unsqueeze(-1), token_weights).reshape(b_, query_num, stat_num * token_num)
        # Reshape values for final weighted sum
        values = values.reshape(b_, stat_num, token_num, d_).reshape(b_, stat_num * token_num, d_)
        return self.W_o(torch.bmm(combine_weights, values))


class MultiHeadAttention(nn.Module):
    """Multi-head attention.
    """

    def __init__(self, query_size, key_size, value_size, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        # Linear layers for projecting queries, keys, and values
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        # Project and reshape queries, keys, and values for multiple heads
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        # Repeat valid_lens for each head
        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        # Compute attention outputs
        output = self.attention(queries, keys, values, valid_lens)

        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


class DotProductAttention(nn.Module):
    """
    Scaled dot product attention.
    """

    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # Compute scaled attention scores
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        # Apply masked softmax to handle valid lengths
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


class EncoderBlock(nn.Module):
    """
    Encoder block with self-attention and feed-forward layers.
    """
    def __init__(self, d_model, d_ff, head_num, dropout=0.1):
        super(EncoderBlock, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, d_model, d_model, d_model, head_num, dropout)
        self.feedForward = PositionWiseFFN(d_model, d_ff, dropout)
        self.add_norm1 = AddNorm(d_model)
        self.add_norm2 = AddNorm(d_model)

    def forward(self, x, valid_lens):
        y = self.self_attention(x, x, x, valid_lens)
        y = self.add_norm1(x, y)
        z = self.feedForward(y)
        z = self.add_norm2(y, z)
        return z


class Encoder(nn.Module):
    """
    Standard Transformer Encoder with stacked Encoder Blocks.
    """
    def __init__(self, d_model, d_ff, head_num, N=6, dropout=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        # Stack N EncoderBlocks into a ModuleList
        self.layers = nn.ModuleList([EncoderBlock(d_model, d_ff, head_num, dropout) for _ in range(N)])

    def forward(self, x, valid_lens):
        # Pass input sequentially through all encoder layers
        for layer in self.layers:
            x = layer(x, valid_lens)

        return x


class EncoderBlockWithRPR(nn.Module):
    """
    Encoder block with relative positional representation (RPR).
    """
    def __init__(self, d_model, d_ff, head_num, clipping_distance, dropout=0.1):
        super(EncoderBlockWithRPR, self).__init__()
        # Multi-head attention with RPR
        self.self_attention = MultiHeadAttentionWithRPR(d_model, d_model, d_model, d_model, head_num,
                                                        clipping_distance, dropout)
        self.feedForward = PositionWiseFFN(d_model, d_ff, dropout)
        # Add and Normalize layers for residual connections
        self.add_norm1 = AddNorm(d_model)
        self.add_norm2 = AddNorm(d_model)

    def forward(self, x, valid_lens):
        y = self.self_attention(x, x, x, valid_lens)
        y = self.add_norm1(x, y)
        z = self.feedForward(y)
        z = self.add_norm2(y, z)
        return z


class EncoderWithRPR(nn.Module):
    """
    Transformer Encoder with RPR.
    """
    def __init__(self, d_model, d_ff, head_num, clipping_distance, N=6, dropout=0.1):
        super(EncoderWithRPR, self).__init__()
        self.d_model = d_model
        # Stack N EncoderBlockWithRPR layers
        self.layers = nn.ModuleList(
            [EncoderBlockWithRPR(d_model, d_ff, head_num, clipping_distance, dropout) for _ in range(N)])

    def forward(self, x, valid_lens):
        # Pass input sequentially through all encoder layers
        for layer in self.layers:
            x = layer(x, valid_lens)

        return x


class DecoderBlock_RL(nn.Module):
    """
    Decoder block for reinforcement learning-based architecture.
    """
    def __init__(self, i, d_model, d_intent, d_ff, head_num, stat_k, token_k, dropout=0.1):
        super(DecoderBlock_RL, self).__init__()
        self.i = i
        # Masked self-attention for autoregressive decoding
        self.masked_attention = MultiHeadAttention(d_model, d_model, d_model, d_model, head_num, dropout)
        # Selective attention for incorporating hierarchical knowledge
        self.selective_attention = SelectiveAttention(d_model + d_intent, d_model, d_model, d_model,
                                                      stat_k, token_k)
        self.cross_attention_exemplar = MultiHeadAttention(d_model + d_intent, d_model, d_model, d_model, head_num, dropout)
        self.gate = nn.Linear(d_model + d_model, 1, bias=False)
        # Position-wise feed-forward network
        self.feedForward = PositionWiseFFN(d_model, d_ff, dropout)
        # Add and Normalize layers for residual connections
        self.add_norm1 = AddNorm(d_model)
        self.add_norm2 = AddNorm(d_model)
        self.add_norm3 = AddNorm(d_model)

    def forward(self, x, state):
        # Extract elements from the state
        stat_enc, stat_valid_len = state[0], state[1]
        exemplar_enc, example_valid_len = state[3], state[4]
        stat_feature, intent_embed = state[-2], state[-1]
        # Maintain a running memory of key-values for masked self-attention
        if state[2][self.i] is None:
            key_values = x
        else:
            key_values = torch.cat((state[2][self.i], x), axis=1)
        state[2][self.i] = key_values
        # Create valid lengths for autoregressive decoding
        if self.training:
            batch_size, num_steps, _ = x.shape
            dec_valid_lens = torch.arange(1, num_steps + 1, device=x.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None

        # Masked self-attention
        x2 = self.masked_attention(x, key_values, key_values, dec_valid_lens)
        y = self.add_norm1(x, x2)
         # Combine current output with intent embedding
        y_intent = torch.cat([y, intent_embed.repeat(1, y.size(1), 1)], dim=-1)
        y2_select = self.selective_attention(y_intent, stat_feature, stat_enc, stat_enc, stat_valid_len)
        y2_exemplar = self.cross_attention_exemplar(y_intent, exemplar_enc, exemplar_enc, example_valid_len)

        # Combine selective and exemplar outputs using a gating mechanism
        gate_weight = torch.sigmoid(self.gate(torch.cat([y2_select, y2_exemplar], dim=-1)))
        y2 = gate_weight * y2_select + (1. - gate_weight) * y2_exemplar
        # Apply normalization and residual connections
        z = self.add_norm2(y, y2 * 2)
        return self.add_norm3(z, self.feedForward(z)), state


class Decoder(nn.Module):
    """
    Decoder with reinforcement learning-based architecture.
    """
    def __init__(self, vocab_size, d_model, d_intent, d_ff, head_num, stat_k, token_k, N=6, dropout=0.1):
        super(Decoder, self).__init__()
        self.num_layers = N
        self.d_model = d_model
        # Stack N DecoderBlock_RL layers
        self.layers = nn.ModuleList(
            [DecoderBlock_RL(i, d_model, d_intent, d_ff, head_num, stat_k, token_k, dropout) for i in
             range(self.num_layers)])
        # Linear layer for final vocabulary projection
        self.dense = nn.Linear(d_model + d_intent, vocab_size)

    def init_state(self, stat_enc, stat_valid_len, exemplar_enc, example_valid_len, stat_feature, intent_embed):
        # Initialize state for decoding
        return [stat_enc, stat_valid_len, [None] * self.num_layers, exemplar_enc, example_valid_len,
                stat_feature, intent_embed]

    def forward(self, x, state, intent_embed):
        # Pass input through each decoder layer
        for layer in self.layers:
            x, state = layer(x, state)
        return self.dense(torch.cat([x, intent_embed.repeat(1, x.size(1), 1)], dim=-1)), state


class Generator(nn.Module):
    def __init__(self, d_model, d_intent, d_ff, head_num, enc_layer_num, dec_layer_num, vocab_size, max_comment_len,
                 clip_dist_code, eos_token, intent_num, stat_k, token_k, dropout=0.1, beam_width=None):
        super(Generator, self).__init__()
        # Embedding layers for shared vocabulary, intents, and positional encodings for comments
        self.share_embedding = nn.Embedding(vocab_size, d_model)
        self.intent_embedding = nn.Embedding(intent_num, d_intent)
        self.comment_pos_embedding = nn.Embedding(max_comment_len + 2, d_model)

        # Encoders for code and exemplars, along with a decoder for generating comments
        self.code_encoder = EncoderWithRPR(d_model, d_ff, head_num, clip_dist_code, enc_layer_num, dropout)
        self.exemplar_encoder = Encoder(d_model, d_ff, head_num, enc_layer_num, dropout)
        self.decoder = Decoder(vocab_size, d_model, d_intent, d_ff, head_num, stat_k, token_k, dec_layer_num, dropout)

         # Dropout for regularization and initialization of key hyperparameters
        self.dropout = nn.Dropout(dropout)
        self.eos_token = eos_token
        self.d_model = d_model
        self.beam_width = beam_width
        self.max_comment_len = max_comment_len
        self.dec_layer_num = dec_layer_num

    def forward(self, code, exemplar, comment, stat_valid_len, exemplar_valid_len, intent):
        """
        Forward pass for generating comments.
        """
        # Encode code input using shared embeddings and the code encoder
        code_embed = self.dropout(self.share_embedding(code))
        code_enc = self.code_encoder(code_embed, None)[:, 1:, :]

        # Prepare state features from the encoded code
        b_, _, d_ = code_enc.size()
        stat_num_inbatch = max(stat_valid_len)
        stat_enc = code_enc.reshape(b_, stat_num_inbatch, -1, d_)
        stat_feature = torch.max(stat_enc, dim=2)[0]
        
        # Embed intent and encode exemplars
        intent_embed = self.intent_embedding(intent.view(-1, 1))
        b_, r_exemplar = exemplar.size()
        exemplar_pos = torch.arange(1, r_exemplar + 1, device=exemplar.device).repeat(b_, 1)
        exemplar_embed = self.dropout(self.share_embedding(exemplar) + self.comment_pos_embedding(exemplar_pos))
        exemplar_enc = self.exemplar_encoder(exemplar_embed, exemplar_valid_len)

        dec_state = self.decoder.init_state(stat_enc.reshape(b_ * stat_num_inbatch, -1, d_), stat_valid_len,
                                            exemplar_enc, exemplar_valid_len, stat_feature, intent_embed)

        # Training mode: Predict tokens based on ground-truth comments
        if self.training:
            r_comment = comment.size(1)
            comment_pos = torch.arange(r_comment, device=comment.device).repeat(b_, 1)
            comment_embed = self.dropout(self.share_embedding(comment) + self.comment_pos_embedding(comment_pos))
            comment_pred, state = self.decoder(comment_embed, dec_state, intent_embed)
            return comment_pred
        else:
            # Generate comments via greedy or beam search depending on settings
            if self.beam_width is None:
                return self.greed_search(b_, comment, dec_state)
            else:
                return self.beam_search(b_, comment, dec_state, self.beam_width, stat_num_inbatch)

    def greed_search(self, batch_size, comment, dec_state):
        # Greedy search for comment generation
        comment_pred = [[-1] for _ in range(batch_size)]
        intent_embed = dec_state[-1]
        # Iterate over positions to predict tokens one-by-one
        for pos_idx in range(self.max_comment_len):
            pos = torch.arange(pos_idx, pos_idx + 1, device=comment.device).repeat(batch_size, 1)
            comment_embed = self.dropout(self.share_embedding(comment) + self.comment_pos_embedding(pos))
            tensor, dec_state = self.decoder(comment_embed, dec_state, intent_embed)
            # Choose the most probable token at each step
            comment = torch.argmax(tensor, -1).detach()
            for i in range(batch_size):
                if comment_pred[i][-1] != self.eos_token:
                    comment_pred[i].append(int(comment[i]))

        comment_pred = [x[1:-1] if x[-1] == self.eos_token and len(x) > 2 else x[1:] for x in comment_pred]
        return comment_pred

    def beam_search(self, batch_size, comment, dec_state, beam_width, stat_num_inbatch):
        # Beam search for more diverse and optimal comment generation
        node_list = []
        batchNode_dict = {i: None for i in range(beam_width)}
        # Initialize nodes for each example in the batch  
        for batch_idx in range(batch_size):
            node_comment = comment[batch_idx].unsqueeze(0)
            node_dec_state = [dec_state[0][batch_idx * stat_num_inbatch:(batch_idx+1) * stat_num_inbatch],
                              dec_state[1][batch_idx].unsqueeze(0),
                              [None] * self.dec_layer_num,
                              dec_state[3][batch_idx].unsqueeze(0), dec_state[4][batch_idx].unsqueeze(0),
                              dec_state[5][batch_idx].unsqueeze(0), dec_state[6][batch_idx].unsqueeze(0)]
            node_list.append(BeamSearchNode(node_dec_state, None, node_comment, 0, 0))
        batchNode_dict[0] = BatchNode(node_list)

        pos_idx = 0
        while pos_idx < self.max_comment_len:
            # Priority queues for handling multiple beam paths
            beamNode_dict = {i: PriorityQueue() for i in range(batch_size)}
            count = 0
            # Explore nodes at each beam level
            for idx in range(beam_width):
                if batchNode_dict[idx] is None:
                    continue

                batchNode = batchNode_dict[idx]
                comment = batchNode.get_comment()
                dec_state = batchNode.get_dec_state()

                # Generate predictions for current position
                pos = torch.arange(pos_idx, pos_idx + 1, device=comment.device).repeat(batch_size, 1)
                comment = self.dropout(self.share_embedding(comment) + self.comment_pos_embedding(pos))
                tensor, dec_state = self.decoder(comment, dec_state, dec_state[-1])
                tensor = F.log_softmax(tensor.squeeze(1), -1).detach()
                log_prob, comment_candidates = torch.topk(tensor, beam_width, dim=-1)

                # Create new nodes for each candidate
                for batch_idx in range(batch_size):
                    pre_node = batchNode.list_node[batch_idx]
                    node_dec_state = [dec_state[0][batch_idx * stat_num_inbatch:(batch_idx+1) * stat_num_inbatch],
                                      dec_state[1][batch_idx].unsqueeze(0),
                                      [l[batch_idx].unsqueeze(0) for l in dec_state[2]],
                                      dec_state[3][batch_idx].unsqueeze(0), dec_state[4][batch_idx].unsqueeze(0),
                                      dec_state[5][batch_idx].unsqueeze(0), dec_state[6][batch_idx].unsqueeze(0)]
                    # Handle end-of-sequence token
                    if pre_node.history_word[-1] == self.eos_token:
                        new_node = BeamSearchNode(node_dec_state, pre_node.prevNode, pre_node.commentID,
                                                  pre_node.logp, pre_node.leng)
                        assert new_node.score == pre_node.score
                        assert new_node.history_word == pre_node.history_word
                        beamNode_dict[batch_idx].put((-new_node.score, count, new_node))
                        count += 1
                        continue
                    # Expand beam paths for each candidate
                    for beam_idx in range(beam_width):
                        node_comment = comment_candidates[batch_idx][beam_idx].view(1, -1)
                        node_log_prob = log_prob[batch_idx][beam_idx].item()
                        new_node = BeamSearchNode(node_dec_state, pre_node, node_comment, pre_node.logp + node_log_prob,
                                                  pre_node.leng + 1)
                        beamNode_dict[batch_idx].put((-new_node.score, count, new_node))
                        count += 1
            # Consolidate top nodes for the next iteration
            for beam_idx in range(beam_width):
                node_list = [beamNode_dict[batch_idx].get()[-1] for batch_idx in range(batch_size)]
                batchNode_dict[beam_idx] = BatchNode(node_list)

            pos_idx += 1
        best_node = batchNode_dict[0]
        comment_pred = []
        for batch_idx in range(batch_size):
            history_word = best_node.list_node[batch_idx].history_word
            if history_word[-1] == self.eos_token and len(history_word) > 2:
                comment_pred.append(history_word[1:-1])
            else:
                comment_pred.append(history_word[1:])

        return comment_pred

    def beam_search_oneExample(self, batch_size, comment, dec_state, beam_width):
        assert batch_size == 1
        # Initialize the first beam search node with the given decoding state and comment.
        node_count = 0
        node = BeamSearchNode(dec_state, None, comment, 0, 0)
        nodes_list = [(-node.score, node_count, node)]
        node_count += 1

        pos_idx = 0
        while pos_idx < self.max_comment_len:

            all_nodes_queue = PriorityQueue()

            for idx in range(len(nodes_list)):
                pre_score, _, pre_node = nodes_list[idx]
                comment = pre_node.commentID
                dec_state = pre_node.dec_state
                # If the previous node has ended with the <eos> token, keep it in the queue.
                if pre_node.history_word[-1] == self.eos_token:
                    all_nodes_queue.put((pre_score, _, pre_node))
                    continue
                
                # Compute embeddings for the next position.
                pos = torch.arange(pos_idx, pos_idx + 1, device=comment.device).repeat(batch_size, 1)
                comment = self.dropout(self.share_embedding(comment) + self.comment_pos_embedding(pos))
                # Decode the next token and get top-k candidates.
                tensor, dec_state = self.decoder(comment, dec_state, dec_state[-1])
                tensor = F.log_softmax(tensor.squeeze(1), -1).detach()
                log_prob, comment_candidates = torch.topk(tensor, beam_width, dim=-1)

                # Create new beam search nodes for the top-k candidates.
                for beam_idx in range(beam_width):
                    node_log_prob = log_prob[0][beam_idx].item()
                    node_comment = comment_candidates[0][beam_idx].view(1, -1)
                    new_node = BeamSearchNode(dec_state, pre_node, node_comment, pre_node.logp + node_log_prob,
                                              pre_node.leng + 1)
                    all_nodes_queue.put((-new_node.score, node_count, new_node))
                    node_count += 1

            # Select the best `beam_width` nodes for the next iteration.
            range_num = min(beam_width, all_nodes_queue.qsize())
            temp_nodes_list = []
            Flag = True
            for _ in range(range_num):
                cur_score, cur_count, cur_node = all_nodes_queue.get()
                if cur_node.history_word[-1] != self.eos_token:
                    Flag = False # Not all nodes have ended with <eos>.
                temp_nodes_list.append((cur_score, cur_count, cur_node))
            nodes_list = temp_nodes_list
            pos_idx += 1
            if Flag: # Terminate early if all nodes end with <eos>.
                break
        
        # Retrieve the best node's predicted sequence.
        best_score, _, best_node = nodes_list[0]
        comment_pred = []
        for batch_idx in range(batch_size):
            history_word = best_node.history_word
            if history_word[-1] == self.eos_token and len(history_word) > 2:
                comment_pred.append(history_word[1:-1]) # Remove <start> and <eos>.
            else:
                comment_pred.append(history_word[1:]) # Remove only <start>.

        return comment_pred # Final predicted comment sequence.


class BatchNode(object):
    def __init__(self, list_node):
        self.list_node = list_node

    def get_comment(self):
        # Concatenates comment IDs from all nodes into a single tensor for batch processing.
        comment_list = [node.commentID for node in self.list_node]
        return torch.cat(comment_list, dim=0)

    def get_dec_state(self):
        # Collects and concatenates decoder states from all nodes in the batch.
        dec_state_list = [node.dec_state for node in self.list_node]
        batch_dec_state = [torch.cat([batch_state[0] for batch_state in dec_state_list], dim=0),
                           torch.cat([batch_state[1] for batch_state in dec_state_list], dim=0)]
        if dec_state_list[0][2][0] is None: # If there's no additional state, replicate the default structure.
            batch_dec_state.append(dec_state_list[0][2])
        else:
            state_3 = []
            for i in range(len(dec_state_list[0][2])):
                state_3.append(torch.cat([batch_state[2][i] for batch_state in dec_state_list], dim=0))
            assert len(state_3) == len(dec_state_list[0][2])
            batch_dec_state.append(state_3)

        # Concatenate other parts of the decoder state.
        batch_dec_state.extend([torch.cat([batch_state[3] for batch_state in dec_state_list], dim=0),
                                torch.cat([batch_state[4] for batch_state in dec_state_list], dim=0),
                                torch.cat([batch_state[5] for batch_state in dec_state_list], dim=0),
                                torch.cat([batch_state[6] for batch_state in dec_state_list], dim=0)])
        return batch_dec_state

    def if_allEOS(self, eos_token):
        # Check if all nodes in the batch have ended with <eos>.
        for node in self.list_node:
            if node.history_word[-1] != eos_token:
                return False
        return True


class BeamSearchNode(object):
    def __init__(self, dec_state, previousNode, commentID, logProb, length, length_penalty=0.75):
        # Store the decoder state, previous node, comment ID, log probability, and length.
        self.dec_state = dec_state
        self.prevNode = previousNode
        self.commentID = commentID
        self.logp = logProb
        self.leng = length
        self.length_penalty = length_penalty
        # If this node has no parent (it's the root node), initialize its history and score.
        if self.prevNode is None:
            self.history_word = [int(commentID)]
            self.score = -100
        else:
            # For non-root nodes, append the current word to the parent's history of words.
            self.history_word = previousNode.history_word + [int(commentID)]
            self.score = self.eval()

    def eval(self):
        '''
        Evaluates the score of the current node by normalizing the log probability
        using a length penalty.
        '''
        # Compute the score by dividing log probability by the length raised to the length penalty power.
        return self.logp / self.leng ** self.length_penalty
