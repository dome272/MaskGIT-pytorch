import math
import torch
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if "Linear" in classname or "Embedding" == classname:
        print(f"Initializing Module {classname}.")
        nn.init.trunc_normal_(m.weight.data, 0.0, 0.02)
    # elif "Parameter" in classname:
    #     return nn.init.trunc_normal_(m, 0.0, 0.02)


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


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class Encoder(nn.Module):
    """
    Transformer encoder using MultiHeadAttention and MLP along with skip connections and LayerNorm
    """
    def __init__(self, dim=768, hidden_dim=3072):
        super(Encoder, self).__init__()
        # self.MultiHeadAttention = MultiHeadAttention(dim)
        self.MultiHeadAttention = nn.MultiheadAttention(dim, num_heads=8, batch_first=True, dropout=0.1)
        self.LayerNorm1 = nn.LayerNorm(dim, eps=1e-12)
        self.LayerNorm2 = nn.LayerNorm(dim, eps=1e-12)
        self.MLP = nn.Sequential(*[
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=0.1)
        ])
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        # attn = self.MultiHeadAttention(x)
        attn, _ = self.MultiHeadAttention(x, x, x, need_weights=False)
        attn = self.dropout(attn)
        x = x.add(attn)
        x = self.LayerNorm1(x)
        mlp = self.MLP(x)
        x = x.add(mlp)
        x = self.LayerNorm2(x)
        return x


class BidirectionalTransformer(nn.Module):
    def __init__(self, args):
        super(BidirectionalTransformer, self).__init__()
        self.num_image_tokens = args.num_image_tokens
        self.tok_emb = nn.Embedding(args.num_codebook_vectors + 2, args.dim)
        # self.pos_emb = PositionalEmbedding(args.dim, self.num_image_tokens + 1)
        self.pos_emb = nn.init.trunc_normal_(nn.Parameter(torch.zeros(self.num_image_tokens + 1, args.dim)), 0., 0.02)
        # self.register_buffer("pos_emb", nn.init.trunc_normal_(nn.Parameter(torch.zeros(1024, args.dim)), 0., 0.02))
        self.blocks = nn.Sequential(*[Encoder(args.dim, args.hidden_dim) for _ in range(args.n_layers)])
        self.Token_Prediction = nn.Sequential(*[
            nn.Linear(in_features=args.dim, out_features=args.dim),
            nn.GELU(),
            nn.LayerNorm(args.dim, eps=1e-12)
        ])
        self.bias = nn.Parameter(torch.zeros(self.num_image_tokens+1, args.num_codebook_vectors + 2))
        self.ln = nn.LayerNorm(args.dim, eps=1e-12)
        self.drop = nn.Dropout(p=0.1)
        self.apply(weights_init)

    def forward(self, x):
        token_embeddings = self.tok_emb(x)
        t = token_embeddings.shape[1]
        position_embeddings = self.pos_emb[:t, :]
        # position_embeddings = self.pos_emb(x)
        embed = self.drop(self.ln(token_embeddings + position_embeddings))
        embed = self.blocks(embed)
        embed = self.Token_Prediction(embed)
        logits = torch.matmul(embed, self.tok_emb.weight.T) + self.bias

        return logits
