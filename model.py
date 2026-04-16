import os
os.environ["DGL_USE_GRAPHBOLT"] = "0"
import dgl
from dgl.nn.pytorch import RelGraphConv
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. Autoencoder (Text Compression) ---
class TextEmbeddingAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim, dropout_rate=0.2):
        super(TextEmbeddingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim * 2),
            nn.BatchNorm1d(encoding_dim * 2),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(encoding_dim * 2, encoding_dim),
            nn.BatchNorm1d(encoding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, encoding_dim * 2),
            nn.BatchNorm1d(encoding_dim * 2),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(encoding_dim * 2, input_dim),
            nn.ReLU(True)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class BaseRGCN(nn.Module):
    def __init__(self, num_nodes, hidden_dim, output_dim, num_relations, num_bases=-1, 
                 num_hidden_layers=1, dropout=0.0, use_self_loop=False, use_cuda=False, 
                 pretrained_text_embeddings=None, pretrained_domain_embeddings=None, 
                 freeze=False, w_text=0.5, w_domain=0.5):
        super(BaseRGCN, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_relations = num_relations
        self.num_bases = None if num_bases < 0 else num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop
        self.use_cuda = use_cuda
        self.pretrained_text_embeddings = pretrained_text_embeddings
        self.pretrained_domain_embeddings = pretrained_domain_embeddings
        self.freeze = freeze
        
        # Dual-Stream Weights
        self.w_text = w_text
        self.w_domain = w_domain

        self.build_model()

    def build_model(self):
        self.layers = nn.ModuleList()
        input_layer = self.build_input_layer()
        if input_layer is not None:
            self.layers.append(input_layer)
        for idx in range(self.num_hidden_layers):
            hidden_layer = self.build_hidden_layer(idx)
            self.layers.append(hidden_layer)

    def build_input_layer(self):
        return None

    def build_hidden_layer(self, idx):
        raise NotImplementedError

    def forward(self, graph, node_ids, rel_ids, norm):
        x = node_ids
        for layer in self.layers:
            x = layer(graph, x, rel_ids, norm)
        return x


# --- 2. GRASP Fusion Layer ---
class GRASP_Fusion(nn.Module):
    """
    Concept:
    - Norm in Hyperbolic Space = Structural Uncertainty Proxy
    - Hub (Low Norm) -> Low Uncertainty -> Trust Structure
    - Tail (High Norm) -> High Uncertainty -> Trust Text Evidence
    """
    def __init__(self, num_nodes, hidden_dim, pretrained_text_embeddings, 
                 pretrained_domain_embeddings, freeze=False, 
                 w_text=0.5, w_domain=0.5):
        super(GRASP_Fusion, self).__init__()
        
        self.w_text = w_text
        self.w_domain = w_domain
        eps = 1e-8

        # --- (A) Structural Stream (Poincaré) ---
        if pretrained_domain_embeddings is not None:
            domain_np = torch.from_numpy(pretrained_domain_embeddings).float()
            # Raw Embedding for Norm/Uncertainty Calculation
            self.raw_domain_embeddings = nn.Embedding.from_pretrained(domain_np, freeze=True) 
            
            # Normalized Feature Embedding
            denom = (domain_np.max() - domain_np.min()) + eps
            norm_domain = (domain_np - domain_np.min()) / denom
            self.emb_domain = nn.Embedding.from_pretrained(norm_domain, freeze=freeze)
            
            # Identity Init Projection (Geometry Preservation)
            input_dim = pretrained_domain_embeddings.shape[1]
            self.proj_domain = nn.Linear(input_dim, hidden_dim)
            if input_dim == hidden_dim:
                nn.init.eye_(self.proj_domain.weight)
                nn.init.zeros_(self.proj_domain.bias)
        else:
            self.raw_domain_embeddings = nn.Embedding(num_nodes, hidden_dim) # Placeholder
            self.emb_domain = nn.Embedding(num_nodes, hidden_dim)
            self.proj_domain = nn.Linear(hidden_dim, hidden_dim)
            nn.init.eye_(self.proj_domain.weight)

        # --- (B) Semantic Stream (Text) ---
        if pretrained_text_embeddings is not None:
            text_np = torch.from_numpy(pretrained_text_embeddings).float()
            denom = (text_np.max() - text_np.min()) + eps
            norm_text = (text_np - text_np.min()) / denom
            self.emb_text = nn.Embedding.from_pretrained(norm_text, freeze=freeze)
            self.autoencoder = TextEmbeddingAutoencoder(pretrained_text_embeddings.shape[1], hidden_dim)
        else:
            self.emb_text = nn.Embedding(num_nodes, hidden_dim)
            self.autoencoder = TextEmbeddingAutoencoder(hidden_dim, hidden_dim)

        # --- (C) Uncertainty Estimator (The Core) ---
        # Maps Poincaré Norm -> Uncertainty Score (u) -> Text Attention
        self.uncertainty_net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid() # 0 (Certain) ~ 1 (Uncertain)
        )
        
        self.dropout = nn.Dropout(0.2)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, graph, node_ids, rel_ids, norm):
        node_ids = node_ids.long()

        # 1. Feature Extraction
        # Structure Feature
        d_emb = self.proj_domain(self.emb_domain(node_ids.squeeze()))
        
        # Text Feature (Encoded)
        t_emb_raw = self.emb_text(node_ids.squeeze())
        t_emb, _ = self.autoencoder(t_emb_raw)
        
        # 2. Uncertainty Estimation from Geometry
        # Calculate Norm from Raw Poincaré (Pretrained Topology)
        raw_d = self.raw_domain_embeddings(node_ids.squeeze())
        geo_norm = torch.norm(raw_d, dim=-1, keepdim=True) 
        
        # Uncertainty Score: High Norm (Tail) -> High Uncertainty -> Need More Text
        uncertainty = self.uncertainty_net(geo_norm) 

        # 3. Evidential Fusion (Simulated)
        # Instead of Dempster-Shafer complex rules, we use 'uncertainty' as a modulation gate
        # which is mathematically cleaner and more stable for GNNs.
        
        # Base: Weighted Sum (Prior Belief)
        base_fusion = (self.w_domain * d_emb) + (self.w_text * t_emb)
        
        # Modulation: Inject Text Evidence proportional to Uncertainty
        # If Uncertainty is High (Tail) -> Add more Text Evidence
        # If Uncertainty is Low (Hub) -> Stick to Structure (Base)
        final_embedding = base_fusion + (uncertainty * t_emb)

        final_embedding = self.layer_norm(final_embedding)
        final_embedding = self.dropout(final_embedding)

        return final_embedding


class RGCN(BaseRGCN):
    def build_input_layer(self):
        return GRASP_Fusion(
            self.num_nodes,
            self.hidden_dim,
            self.pretrained_text_embeddings,
            self.pretrained_domain_embeddings,
            freeze=self.freeze,
            w_text=self.w_text,
            w_domain=self.w_domain
        )

    def build_hidden_layer(self, idx):
        activation = F.relu if idx < self.num_hidden_layers - 1 else None
        return RelGraphConv(
            in_feat=self.hidden_dim,
            out_feat=self.hidden_dim,
            num_rels=self.num_relations,
            regularizer='bdd',
            num_bases=self.num_bases,
            activation=activation,
            self_loop=self.use_self_loop,
            dropout=self.dropout
        )


class LinkPredict(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, num_relations, num_bases=-1,
        num_hidden_layers=1, dropout=0.0, use_cuda=False, regularization_param=0.0,
        pretrained_text_embeddings=None, pretrained_domain_embeddings=None,
        pretrained_relation_embeddings=None, 
        freeze=False, w_text=0.5, w_domain=0.5
    ):
        super(LinkPredict, self).__init__()
        self.rgcn = RGCN(
            input_dim, hidden_dim, hidden_dim, num_relations * 2, num_bases,
            num_hidden_layers, dropout, use_cuda,
            pretrained_text_embeddings=pretrained_text_embeddings,
            pretrained_domain_embeddings=pretrained_domain_embeddings,
            freeze=freeze,
            w_text=w_text, w_domain=w_domain
        )
        self.regularization_param = regularization_param

        if pretrained_relation_embeddings is not None:
            self.relation_weights = nn.Parameter(torch.Tensor(pretrained_relation_embeddings))
            normalized_relations = (self.relation_weights - self.relation_weights.min()) / (
                (self.relation_weights.max() - self.relation_weights.min()) + 1e-8
            )
            self.relation_weights.data.copy_(normalized_relations)
        else:
            self.relation_weights = nn.Parameter(torch.Tensor(num_relations, hidden_dim))
            nn.init.xavier_uniform_(self.relation_weights, gain=nn.init.calculate_gain('relu'))

    def calculate_score(self, embeddings, triplets):
        subject_embeddings = embeddings[triplets[:, 0]]
        relation_embeddings = self.relation_weights[triplets[:, 1]]
        object_embeddings = embeddings[triplets[:, 2]]
        score = torch.sum(subject_embeddings * relation_embeddings * object_embeddings, dim=1)
        return score

    def forward(self, graph, node_ids, rel_ids, norm):
        return self.rgcn(graph, node_ids, rel_ids, norm)

    def regularization_loss(self, embeddings):
        return torch.mean(embeddings.pow(2)) + torch.mean(self.relation_weights.pow(2))

    def get_loss(self, graph, embeddings, triplets, labels):
        score = self.calculate_score(embeddings, triplets)
        prediction_loss = F.binary_cross_entropy_with_logits(score, labels)
        reg_loss = self.regularization_loss(embeddings)
        return prediction_loss + self.regularization_param * reg_loss
