import torch
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if "Linear" in classname:
        nn.init.trunc_normal_(m.weight.data, 0.0, 0.02)


class Attention(nn.Module):
    """
    Simple Self-Attention algorithm. Potential for optimization using a non-quadratic attention mechanism in complexity.
    -> Linformer, Reformer etc.
    """
    def __init__(self, dim=768, heads=8):
        super(Attention, self).__init__()
        d = dim // heads
        self.q, self.k, self.v = nn.Linear(dim, d), nn.Linear(dim, d), nn.Linear(dim, d)
        self.norm = d ** 0.5
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        q, k, v = self.q(x), self.k(x), self.v(x)
        qk = torch.softmax(q @ torch.transpose(k, 1, 2) / self.norm, dim=1)
        qk = self.dropout(qk)
        attn = torch.matmul(qk, v)
        return attn


class MultiHeadAttention(nn.Module):
    """
    Implementation of MultiHeadAttention, splitting it up to multiple Self-Attention layers and concatenating
    the results and subsequently running it through one linear layer of same dimension.
    """
    def __init__(self, dim=768, heads=8):
        super(MultiHeadAttention, self).__init__()
        self.self_attention_heads = nn.ModuleList([Attention(dim, heads) for _ in range(heads)])
        self.projector = nn.Linear(dim, dim)

    def forward(self, x):
        for i, sa_head in enumerate(self.self_attention_heads):
            if i == 0:
                out = sa_head(x)
            else:
                out = torch.cat((out, sa_head(x)), axis=-1)
        out = self.projector(out)
        return out


class Encoder(nn.Module):
    """
    Transformer encoder using MultiHeadAttention and MLP along with skip connections and LayerNorm
    """
    def __init__(self, dim=768, hidden_dim=3072):
        super(Encoder, self).__init__()
        self.MultiHeadAttention = MultiHeadAttention(dim)
        self.LayerNorm1 = nn.LayerNorm(dim)
        self.LayerNorm2 = nn.LayerNorm(dim)
        self.MLP = nn.Sequential(*[
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.GELU()
        ])

    def forward(self, x):
        attn = self.MultiHeadAttention(x)
        x = x.add(attn)
        x = self.LayerNorm1(x)
        mlp = self.MLP(x)
        x = x.add(mlp)
        x = self.LayerNorm2(x)
        return x


class BidirectionalTransformer(nn.Module):
    def __init__(self, N=24, dim=768, codebook_size=1024):
        super(BidirectionalTransformer, self).__init__()

        self.tok_emb = nn.Embedding(codebook_size, dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, 512, dim))
        self.EncoderLayers = nn.ModuleList([Encoder(dim) for _ in range(N)])
        self.Token_Prediction = nn.Linear(in_features=dim, out_features=codebook_size)
        self.apply(weights_init)

    def forward(self, x):
        token_embeddings = self.tok_emb(x)
        t = token_embeddings.shape[1]
        position_embeddings = self.pos_emb[:, :t, :]
        embed = token_embeddings + position_embeddings
        for enc_layer in self.EncoderLayers:
            embed = enc_layer(embed)
        tkn_prd = self.Token_Prediction(embed)
        return tkn_prd
