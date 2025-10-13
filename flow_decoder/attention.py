import torch
import torch.nn as nn
import math
from typing import Tuple, Optional

class AttentionHead(nn.Module):
    """Single-head self-attention."""
    def __init__(self, dim: int, n_hidden: int) -> None:
        """
        Args:
            dim: Feature width of each latent token.
            n_hidden: Size of internal projections (Q/K/V) and the pooled output.
        """

        super().__init__()

        self.W_K = nn.Linear(dim, n_hidden) # W_K weight matrix
        self.W_Q = nn.Linear(dim, n_hidden) # W_Q weight matrix
        self.W_V = nn.Linear(dim, n_hidden) # W_V weight matrix
        self.n_hidden = n_hidden
        self.scale = math.sqrt(n_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs self-attention on the input sequence.

        Args:
            x: Tensor shaped `[batch, num_tokens, dim]`.

        Returns:
            Attention output `[batch, num_tokens, n_hidden]` and attention weights.
        """
        B, N, D = x.shape
        device = x.device

        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        QK = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        alpha = torch.nn.functional.softmax(QK, dim=-1)

        out = torch.matmul(alpha, V)

        return out, alpha

class MultiHeadedAttention(nn.Module):
    def __init__(self, dim: int, n_hidden: int, num_heads: int):
        # dim: the dimension of the input
        # n_hidden: the hidden dimenstion for the attention layer
        # num_heads: the number of attention heads
        super().__init__()

        self.heads = nn.ModuleList([AttentionHead(dim=dim, n_hidden=n_hidden) for i in range(num_heads)])
        self.linearproj = torch.nn.Linear(in_features = n_hidden * num_heads, out_features = dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x                the inputs. shape: (B x T x dim)
        #
        # Outputs:
        # attn_output      the output of performing multi-headed self-attention on x.
        #                  shape: (B x T x dim)
        # attn_alphas      the attention weights of each of the attention heads.
        #                  shape: (B x Num_heads x T x T)

        attn_output_list=[]
        attn_alphas_list=[]
        for head in self.heads:
          attn_output_head, alpha = head(x)
          attn_output_list.append(attn_output_head)
          attn_alphas_list.append(alpha.unsqueeze(1))

        attn_alphas = torch.cat(attn_alphas_list, dim=1)
        attn_output_raw = torch.cat(attn_output_list, dim=-1)

        attn_output = self.linearproj(attn_output_raw)

        return attn_output, attn_alphas


# these are already implemented for you!

class FFN(nn.Module):
    def __init__(self, dim: int, n_hidden: int):
        # dim       the dimension of the input
        # n_hidden  the width of the linear layer

        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, n_hidden),
            nn.GELU(),
            nn.Linear(n_hidden, dim),
        )

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        # x         the input. shape: (B x T x dim)

        # Outputs:
        # out       the output of the feed-forward network: (B x T x dim)
        return self.net(x)

class AttentionResidual(nn.Module):
    def __init__(self, dim: int, attn_dim: int, mlp_dim: int, num_heads: int):
        # dim       the dimension of the input
        # attn_dim  the hidden dimension of the attention layer
        # mlp_dim   the hidden layer of the FFN
        # num_heads the number of heads in the attention layer
        super().__init__()
        self.attn = MultiHeadedAttention(dim, attn_dim, num_heads)
        self.ffn = FFN(dim, mlp_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x                the inputs. shape: (B x T x dim)
        # attn_mask        an attention mask. If None, ignore. If not None, then mask[b, i, j]
        #                  contains 1 if (in batch b) token i should attend on token j and 0
        #                  otherwise. shape: (B x T x T)
        #
        # Outputs:
        # attn_output      shape: (B x T x dim)
        # attn_alphas      the attention weights of each of the attention heads.
        #                  shape: (B x Num_heads x T x T)

        attn_out, alphas = self.attn(x=x)
        x = attn_out + x
        x = self.ffn(x) + x
        return x, alphas



class TransformerPooling(nn.Module):
    def __init__(self, dim: int, attn_dim: int, mlp_dim: int, num_heads: int, num_layers: int):
        # dim       the dimension of the input
        # attn_dim  the hidden dimension of the attention layer
        # mlp_dim   the hidden layer of the FFN
        # num_heads the number of heads in the attention layer
        # num_layers the number of attention layers.
        super().__init__()

        self.dim=dim
        self.n_hidden=attn_dim
        self.n_heads=num_heads
        self.n_layers=num_layers

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # self.layers = nn.ModuleList([MultiHeadedAttention(dim=dim,n_hidden=attn_dim,num_heads=num_heads)])
        self.layers = nn.ModuleList([AttentionResidual(dim,attn_dim,mlp_dim,num_heads) for i in range(num_layers)])


    def forward(self, x: torch.Tensor, return_attn=False):
        # x                the inputs. shape: (B x T x dim)
        #
        # Outputs:
        # attn_output      shape: (B x dim) - pooled CLS token
        # attn_alphas      If return_attn is False, return None. Otherwise return the attention weights
        #                  of each of each of the attention heads for each of the layers.
        #                  shape: (B x Num_layers x Num_heads x T+1 x T+1)

        B, T, D = x.shape
        device = x.device

        # Add CLS token at the beginning
        cls_token = self.cls_token.expand(B, -1, -1).to(device)
        x_with_cls = torch.cat([cls_token, x], dim=1)  # [B, T+1, D]

        alphas_list=[]
        for layer in self.layers:
          x_with_cls, alphas = layer(x_with_cls)
          alphas_list.append(alphas)

        # Extract CLS token (first token)
        output = x_with_cls[:, 0, :]  # [B, D]

        if return_attn:
          collected_attns = torch.stack(alphas_list,dim=1)
          return output, collected_attns
        else:
          return output





# class AttentionPooling(nn.Module):
#     """Attention-based set pooling followed by projection."""
#     def __init__(self, dim: int, n_hidden: int) -> None:
#         """
#         Args:
#             dim: Output embedding size (matches input token width).
#             n_hidden: Hidden size used inside the attention head.
#         """
#         super().__init__()

#         self.dim=dim
#         self.n_hidden=n_hidden
#         self.head = AttentionHead(dim=dim, n_hidden=n_hidden)
#         self.proj = nn.Linear(n_hidden,dim)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Compresses a token set to a single latent.

#         Args:
#             x: Tensor `[batch, num_tokens, dim]`.

#         Returns:
#             Tensor `[batch, 1, dim]` containing the pooled latent.
#         """

#         x = self.head(x)
#         x_pooled = x[:,:1]
#         x_pooled = self.proj(x_pooled)

#         return x_pooled.squeeze(1) # [batch,dim]
