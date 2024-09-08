# Deep Learning 2023 course, held by Professor Paolo Frasconi - University of Florence, Italy
# Created by Giovanni Colombo - Mat. 7092745
# Dedicated Repository on GitHub at https://github.com/giovancombo/ProteinSecondaryStructurePrediction

import torch
import torch.nn as nn
import numpy as np
import random


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class RelativePositionalEncoding(nn.Module):
    """
    Implement Relative Positional Encoding for the Transformer model: it creates learnable embeddings for
    relative positions in the sequence, which are used to inform the self-attention mechanism about
    positional relationships.

    Attributes:
        head_size (int): Dimension of each attention head.
        max_relative_position (int): Maximum relative distance between sequence elements.
        embd (nn.Parameter): Learnable relative position embeddings.
    """

    def __init__(self, head_size, max_relative_position):
        super().__init__()

        self.head_size = head_size
        self.max_relative_position = max_relative_position
        self.embd = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, self.head_size))
        self.scaling = nn.Parameter(torch.Tensor(1))

        nn.init.xavier_uniform_(self.embd)
        nn.init.constant_(self.scaling, 1.0)

    def forward(self, len_q, len_k):
        device = self.embd.device
        range_q = torch.arange(len_q, device = device)
        range_k = torch.arange(len_k, device = device)

        rel_pos = range_q[None, :] - range_k[:, None]
        # scaled_rel_pos = torch.exp(-torch.abs(rel_pos).float() / (self.max_relative_position * self.scaling))

        rel_pos = torch.clamp(rel_pos, - self.max_relative_position, self.max_relative_position)
        rel_pos += self.max_relative_position
        rel_pos = rel_pos.to(torch.long)            # (len_q, len_k)

        embeddings = self.embd[rel_pos]                   # (len_q, len_k, head_size)

        return embeddings #* scaled_rel_pos.unsqueeze(-1)
    

class MultiHeadAttention(nn.Module):
    """
    This module performs the Multi-Head Attention mechanism of the Transformer, allowing
    the model to attend to different positions with multiple attention heads.
    It includes Relative Positional Encoding for keys and values.

    Attributes:
        d_model (int): Dimension of the model.
        n_heads (int): Number of attention heads.
        head_size (int): Dimension of each attention head.
        fc_q, fc_k, fc_v, fc_o (nn.Linear): Linear projections for queries, keys, values, and output.
        rel_pos_k, rel_pos_v (RelativePositionalEncoding): Relative positional encodings for keys and values.

    Args:
        d_model (int): Dimension of the model.
        n_heads (int): Number of attention heads.
        max_relative_position (int): Maximum relative distance for positional encoding.
        dropout (float): Dropout rate.
        device (torch.device): Device to run the module on.
        seed (int): Random seed for reproducibility.
    """

    def __init__(self, d_model, n_heads, max_relative_position, dropout, device, seed = 42):
        super().__init__()
        set_seed(seed)

        assert d_model % n_heads == 0, "d_model (= embed_dim) must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_size = d_model // n_heads
        self.max_relative_position = max_relative_position

        self.fc_kqv = nn.Linear(d_model, 3 * d_model)
        self.fc_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_size])).to(device)

        self.rel_pos_k = RelativePositionalEncoding(self.head_size, self.max_relative_position)
        self.rel_pos_v = RelativePositionalEncoding(self.head_size, self.max_relative_position)

        for layer in [self.fc_kqv, self.fc_o]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        
    def forward(self, x, mask = None):
        kqv = self.fc_kqv(x)
        query, key, value = kqv.chunk(3, dim = -1)                      # (batch_size, len, d_model)

        batch_size = query.shape[0]
        len_k = key.shape[1]                                            # key = (batch_size, key_len, d_model)
        len_q = query.shape[1]                                          # query = (batch_size, query_len, d_model)
        len_v = value.shape[1]                                          # value = (batch_size, value_len, d_model) = (8,700,49) (example)

        # Computing similarity scores
        rel_q1 = query.view(batch_size, -1, self.n_heads, self.head_size).permute(0, 2, 1, 3)   # (8,7,700,7)
        rel_k1 = key.view(batch_size, -1, self.n_heads, self.head_size).permute(0, 2, 1, 3)     # (8,7,700,7)
        attn_1 = torch.matmul(rel_q1, rel_k1.permute(0, 1, 3, 2))                               # (8,7,700,700)

        # Bias due to relative positional encoding
        rel_q2 = query.permute(1, 0, 2).contiguous().view(len_q, batch_size*self.n_heads, self.head_size)   # (700,56,7)
        rel_k2 = self.rel_pos_k(len_q, len_k)                                                               # (700,700,7)
        attn_2 = torch.matmul(rel_q2, rel_k2.transpose(1, 2)).transpose(0, 1)                               # (56,700,7)
        attn_2 = attn_2.contiguous().view(batch_size, self.n_heads, len_q, len_k)                           # (8,7,700,700)
        
        attn = (attn_1 + attn_2) / self.scale                                                   # (8,7,700,700)

        # Masking
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)                                               # (batch_size, 1, 1, len_k)
            mask = mask.expand(-1, self.n_heads, -1, -1)
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = self.dropout(torch.softmax(attn, dim = -1))                                          # (batch_size, n_heads, query_len, key_len)
        
        rel_v1 = value.view(batch_size, -1, self.n_heads, self.head_size).permute(0, 2, 1, 3)       # (8,7,700,7)
        cxt_1 = torch.matmul(attn, rel_v1)                                                          # (8,7,700,7)

        # Bias due to relative positional encoding
        rel_v2 = self.rel_pos_v(len_q, len_v)                                                       # (700,700,7)                                              
        cxt_2 = attn.permute(2, 0, 1, 3).contiguous().view(len_q, batch_size*self.n_heads, len_k)   # (700,56,700)
        cxt_2 = torch.matmul(cxt_2, rel_v2)                                                         # (700,56,7)
        cxt_2 = cxt_2.transpose(0, 1).contiguous().view(batch_size, self.n_heads, len_q, self.head_size)    # (8,7,700,7)
  
        context = cxt_1 + cxt_2                                         # (batch_size, n_heads, query_len, head_size)
        context = context.permute(0, 2, 1, 3).contiguous()              # (batch_size, query_len, n_heads, head_size)
        context = context.view(batch_size, -1, self.d_model)            # (batch_size, query_len, d_model)
        output = self.fc_o(context)                                     # (batch_size, query_len, d_model)

        return output                                                   # (8, 700, 49)


class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.linear1 = nn.Linear(d_model, 4 * d_model)                 # (batch_size, len, d_model)
        self.linear2 = nn.Linear(4 * d_model, d_model)                 # (batch_size, len, d_model)
        self.relu = nn.ReLU()

        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x):
        x = self.relu(self.linear1(x))                                 # (batch_size, len, 4 * d_model)
        return self.linear2(x)                                         # (batch_size, len, d_model)
    

class EncoderLayer(nn.Module):
    """
    This module combines Multi-Head Self-Attention with a position-wise Feed-Forward Network,
    forming the basic building block of the Transformer Encoder.

    Attributes:
        multi_attn (MultiHeadAttention): Multi-head self-attention layer.
        ffwd (FeedForward): Position-wise feed-forward network.
        layernorm_1, layernorm_2 (nn.LayerNorm): Layer Normalization modules.

    Args:
        d_model (int): Dimension of the model.
        n_heads (int): Number of attention heads.
        max_relative_position (int): Maximum relative distance for positional encoding.
        dropout (float): Dropout rate.
        device (torch.device): Device to run the module on.
        seed (int): Random seed for reproducibility.
    """

    def __init__(self, d_model, n_heads, max_relative_position, dropout, device, seed = 42):
        super().__init__()
        set_seed(seed)

        self.multi_attn = MultiHeadAttention(d_model, n_heads, max_relative_position, dropout, device, seed)        
        self.ffwd = FeedForward(d_model)

        self.layernorm_1 = nn.LayerNorm(d_model)
        self.layernorm_2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask = None):                  # Input: (batch_size, len, d_model)
        # Note: the original paper puts LayerNorms after each sublayer, but it was found that putting them before gives better results
        # There's no Dropout here in the original paper
        
        # attn_input = self.layernorm_1(x)
        # attn_output = self.multi_attn(attn_input, mask)
        # out = x + self.dropout(attn_output)

        # ffwd_input = self.layernorm_2(out)
        # ffwd_output = self.ffwd(ffwd_input)
        # h = out + self.dropout(ffwd_output)

        out = self.layernorm_1(x + self.dropout(self.multi_attn(x, mask)))
        h = self.layernorm_2(out + self.ffwd(out))

        return h                                        # Output: (batch_size, len, d_model)
    

class Encoder(nn.Module):
    """
    This module stacks multiple Encoder layers to form the full Encoder of the Transformer.

    Attributes:
        enc_layers (nn.ModuleList): List of EncoderLayer modules.

    Args:
        d_model (int): Dimension of the model.
        n_heads (int): Number of attention heads.
        n_layers (int): Number of encoder layers.
        max_relative_position (int): Maximum relative distance for positional encoding.
        dropout (float): Dropout rate.
        device (torch.device): Device to run the module on.
        seed (int): Random seed for reproducibility.
    """

    def __init__(self, d_model, n_heads, n_layers, max_relative_position, dropout, device, seed = 42):
        super().__init__()
        set_seed(seed)

        self.enc_layers = nn.ModuleList([EncoderLayer(d_model, n_heads, max_relative_position, dropout, device, seed)
                                         for _ in range(n_layers)])

    def forward(self, x, mask = None):
        for layer in self.enc_layers:
            x = layer(x, mask)
        return x                                        # Output: (batch_size, len, embd_dim + 21 = d_model)
    

class ClassificationHead(nn.Module):
    """
    Classification head for the Transformer model: produces class probabilities for PSSP.

    Attributes:
        clf (nn.Sequential): Sequential container of linear layers and activation functions.

    Args:
        d_model (int): Input dimension from the encoder.
        clf_hid_dim (int): Hidden dimension of the classification head.
        classes (int): Number of output classes.
        dropout (float): Dropout rate.
    """

    def __init__(self, d_model, clf_hid_dim, classes, dropout):
        super().__init__()

        self.clf = nn.Sequential(
            nn.Linear(d_model, clf_hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(clf_hid_dim, classes))

        nn.init.xavier_uniform_(self.clf[0].weight)
        nn.init.zeros_(self.clf[0].bias)
        nn.init.xavier_uniform_(self.clf[-1].weight)
        nn.init.zeros_(self.clf[-1].bias)

    def forward(self, x):
        return self.clf(x)                              # Output: (batch_size, len, classes)
    

class ProteinTransformer(nn.Module):
    """
    Protein Transformer model for PSSP.

    Attributes:
        embd (nn.Embedding): Embedding layer for amino acids.
        encoder (Encoder): Transformer encoder.
        clf (ClassificationHead): Classification head.

    Args:
        vocab_size (int): Number of amino acids.
        embd_dim (int): Dimension of the embedding for the 21 amino acid residues.
        classes (int): Number of output classes: 8 secondary structures + 1 NoSeq class.
        n_heads (int): Number of attention heads.
        n_layers (int): Number of encoder layers.
        max_relative_position (int): Maximum relative distance for relative positional encoding.
        clf_hid_dim (int): Hidden dimension of the classification head.
        dropout (float): Dropout rate.
        temperature (float): Temperature for softmax.
        device (torch.device): Device (CPU or GPU) to run the module on.
        seed (int): Random seed for reproducibility.
    """

    def __init__(self, vocab_size, embd_dim, classes, n_heads, n_layers, max_relative_position, clf_hid_dim, dropout, temperature, device, seed = 42):
        super().__init__()
        set_seed(seed)

        self.embd = nn.Embedding(vocab_size, embd_dim, padding_idx = 20)
        d_model = embd_dim + 21
        self.encoder = Encoder(d_model, n_heads, n_layers, max_relative_position, dropout, device, seed)
        self.clf = ClassificationHead(d_model, clf_hid_dim, classes, dropout)
        self.temperature = temperature

        nn.init.normal_(self.embd.weight, mean=0, std=d_model**-0.5)

    def forward(self, x_amino, x_pssm, mask = None):            # (batch_size, len); (batch_size, len, 21)
        amino_embd = self.embd(x_amino)
        x = torch.cat((amino_embd, x_pssm), -1)                 # (batch_size, len, embd_dim + 21)
        out = self.encoder(x, mask)                             # (batch_size, len, embd_dim + 21)
        logits = self.clf(out)                                  # (batch_size, len, classes)

        return logits / self.temperature
