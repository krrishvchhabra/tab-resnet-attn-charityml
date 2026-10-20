import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit (SwiGLU) for tabular feature filtering.
    """
    def __init__(self, in_features, out_features):
        super(SwiGLU, self).__init__()
        self.fc_val = nn.Linear(in_features, out_features)
        self.fc_gate = nn.Linear(in_features, out_features)

    def forward(self, x):
        # Value * Swish(Gate) where Swish is SiLU
        return self.fc_val(x) * F.silu(self.fc_gate(x))


class SqueezeAndExcitation(nn.Module):
    """
    Squeeze-and-Excitation block for tabular channel/feature recalibration.
    """
    def __init__(self, dim, reduction=4):
        super(SqueezeAndExcitation, self).__init__()
        reduced_dim = max(8, dim // reduction)
        self.se = nn.Sequential(
            nn.Linear(dim, reduced_dim),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_dim, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Learn adaptive weights across latent channels
        scale = self.se(x)
        return x * scale


class AdvancedResidualBlock(nn.Module):
    """
    Phase 2 Enhanced Residual Block integrating:
    1. Pre-Activation (ResNet-v2 style: LayerNorm before branch)
    2. Gated Units (SwiGLU)
    3. Squeeze-and-Excitation (SE channel recalibration)
    4. LayerScale (learnable alpha parameter initialized to 1e-4)
    """
    def __init__(self, dim, dropout=0.2, init_values=1e-4):
        super(AdvancedResidualBlock, self).__init__()
        # 1. Pre-Activation Normalization (using LayerNorm for consistency)
        self.pre_norm = nn.LayerNorm(dim)
        
        # 2. Gated Unit (SwiGLU)
        self.swiglu = SwiGLU(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(dim, dim)

        # 3. Squeeze-and-Excitation
        self.se = SqueezeAndExcitation(dim, reduction=4)

        # 4. LayerScale (learnable alpha parameter)
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        # 1. Pre-Activation branch
        residual = x
        h = self.pre_norm(x)
        
        # 2. Gated transformation
        h = self.swiglu(h)
        h = self.dropout(h)
        h = self.fc_out(h)

        # 3. Squeeze-and-Excitation recalibration
        h = self.se(h)

        # 4. LayerScale + pure identity addition
        return residual + self.gamma * h


class TabularMLP(nn.Module):
    """
    Baseline Deep Learning Model: Multi-Layer Perceptron with Embedding Layers.
    """
    def __init__(self, num_features_dim, cat_dims, hidden_dims=[128, 64], dropout=0.3):
        super(TabularMLP, self).__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_classes, min(50, (num_classes + 1) // 2))
            for num_classes in cat_dims
        ])
        total_emb_dim = sum(emb.embedding_dim for emb in self.embeddings)
        in_dim = num_features_dim + total_emb_dim

        layers = []
        prev_dim = in_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x_num, x_cat):
        emb_outs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat([x_num] + emb_outs, dim=1)
        return self.network(x)


class TabResNetAttn(nn.Module):
    """
    Phase 2 Architecture:
    - Feature Tokenizer: Continuous & Categorical columns projected into shared d-dim tokens.
    - Multi-Head Self-Attention: Pairwise inter-column interaction modeling.
    - Advanced Residual Trunk: Pre-Activation + SwiGLU + SE + LayerScale.
    """
    def __init__(
        self, 
        num_features_dim, 
        cat_dims, 
        token_dim=32, 
        num_heads=4, 
        num_res_blocks=2, 
        dropout=0.2,
        use_attention=True,
        use_residual=True,
        use_embeddings=True,
        init_values=1e-4
    ):
        super(TabResNetAttn, self).__init__()
        self.use_attention = use_attention
        self.use_residual = use_residual
        self.use_embeddings = use_embeddings
        self.num_cols = num_features_dim + len(cat_dims)
        self.token_dim = token_dim

        # Categorical Tokenizers
        if self.use_embeddings:
            self.cat_embeddings = nn.ModuleList([
                nn.Embedding(num_classes, token_dim) for num_classes in cat_dims
            ])
        else:
            self.cat_embeddings = nn.ModuleList([
                nn.Sequential(nn.Linear(num_classes, token_dim)) for num_classes in cat_dims
            ])

        # Numerical Tokenizers
        self.num_tokenizers = nn.ModuleList([
            nn.Linear(1, token_dim) for _ in range(num_features_dim)
        ])

        # Multi-Head Self-Attention
        if self.use_attention:
            self.attn = nn.MultiheadAttention(embed_dim=token_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
            self.attn_norm = nn.LayerNorm(token_dim)

        # Trunk with Phase 2 Advanced Residual Blocks
        flat_dim = self.num_cols * token_dim
        trunk_dim = 128
        self.input_proj = nn.Linear(flat_dim, trunk_dim)

        if self.use_residual:
            self.blocks = nn.ModuleList([
                AdvancedResidualBlock(trunk_dim, dropout=dropout, init_values=init_values)
                for _ in range(num_res_blocks)
            ])
        else:
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(trunk_dim, trunk_dim),
                    nn.BatchNorm1d(trunk_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ) for _ in range(num_res_blocks)
            ])

        self.head = nn.Linear(trunk_dim, 1)

    def forward(self, x_num, x_cat):
        batch_size = x_num.size(0)
        tokens = []

        # Tokenize numerical features
        for i, tokenizer in enumerate(self.num_tokenizers):
            tokens.append(tokenizer(x_num[:, i:i+1]).unsqueeze(1))

        # Tokenize categorical features
        for i, emb in enumerate(self.cat_embeddings):
            if self.use_embeddings:
                tokens.append(emb(x_cat[:, i]).unsqueeze(1))
            else:
                one_hot = F.one_hot(x_cat[:, i], num_classes=emb[0].in_features).float()
                tokens.append(emb(one_hot).unsqueeze(1))

        # [batch_size, num_cols, token_dim]
        x_tokens = torch.cat(tokens, dim=1)

        # Feature Self-Attention
        if self.use_attention:
            attn_out, _ = self.attn(x_tokens, x_tokens, x_tokens)
            x_tokens = self.attn_norm(x_tokens + attn_out)

        # Flatten & pass to Trunk
        h = self.input_proj(x_tokens.view(batch_size, -1))
        for block in self.blocks:
            h = block(h)

        return self.head(h)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)