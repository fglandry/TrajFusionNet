import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import TokenEmbedding, PositionalEmbedding, TemporalEmbedding, TimeFeatureEmbedding
import numpy as np
from utils_tsl.laplacian import get_pe_2d


class Model(nn.Module):
    """
    Vanilla Transformer
    with O(L^2) complexity
    Paper link: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        # Embedding
        self.enc_embedding = DataEmbedding(configs, configs.enc_in, configs.d_model, 
                                           configs.embed, configs.freq, configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast' or self.task_name == 'encoder_decoder_classification':
            self.dec_embedding = DataEmbedding(configs, configs.dec_in, configs.d_model, 
                                               configs.embed, configs.freq, configs.dropout)
            self.decoder = Decoder(
                [
                    DecoderLayer(
                        AttentionLayer(
                            FullAttention(True, configs.factor, attention_dropout=configs.dropout,
                                          output_attention=False),
                            configs.d_model, configs.n_heads),
                        AttentionLayer(
                            FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                          output_attention=False),
                            configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation,
                    )
                    for l in range(configs.d_layers)
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_model),
                projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
            )
        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)
        if self.task_name == 'encoding':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
        if self.task_name == 'encoder_decoder_classification':
            self.projection = nn.Linear(configs.seq_len * configs.c_out, configs.num_class)
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)

    def encoder_decoder_classification(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)

        # Output
        output = self.act(dec_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        if x_mark_enc:
            output = output * x_mark_enc.unsqueeze(-1)  # zero-out padding embeddings
        output = output.reshape(output.shape[0], self.configs.seq_len * self.configs.c_out)
        output = self.projection(output)  # (batch_size, num_classes)
        return output
    
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out)
        return dec_out

    def anomaly_detection(self, x_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out)
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        if x_mark_enc:
            output = output * x_mark_enc.unsqueeze(-1)  # zero-out padding embeddings
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output
    
    def encoding(self, x_enc, x_mark_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        if x_mark_enc:
            output = output * x_mark_enc.unsqueeze(-1)  # zero-out padding embeddings
        #output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        #output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, 
                x_enc, 
                x_mark_enc=None, 
                x_dec=None, 
                x_mark_dec=None, 
                mask=None):
        if self.task_name == 'encoder_decoder_classification':
            dec_out = self.encoder_decoder_classification(x_enc, x_mark_enc, x_dec, x_mark_dec)
            #return dec_out[:, -self.pred_len:, :]  # [B, L, D]
            return dec_out
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        if self.task_name == 'encoding':
            dec_out = self.encoding(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
    

class DataEmbedding(nn.Module):
    def __init__(self, config, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.config = config
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.node_type_embedding = NodeTypeEmbedding(config, d_model=d_model)
        self.laplacian_embedding = LaplacianEmbedding(config, d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            # Calculate type identifiers (tokengt)
            type_identifiers = self.node_type_embedding(x)

            # Calculate node identifiers (tokengt)
            node_identifiers = self.laplacian_embedding(x)

            x = self.value_embedding(x) + type_identifiers + node_identifiers # + self.position_embedding(x)
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)

# Hardcoded values for pedestrian-centric graph
EDGE_NODE_VALS = {
    "scene_graph": {
        "edge_indices": [[0, 0], [1, 1], [2, 2], # scene_graph_nodes  
                        [0, 1], [0, 2]], # scene_graph_edges
        "node_num": 3,
        "edge_num": 2
    },
    "scene_graph_2": {
        "edge_indices": [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4],
                        [5, 5], [6, 6], [7, 7], # scene_graph_nodes  
                        [0, 1], [0, 2],
                        [0, 3], [0, 4], [0, 5], [0, 6], [0, 7]], # scene_graph_edges
        "node_num": 8,
        "edge_num": 7
    },
    "pedestrian_graph": {
        "edge_indices": [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], 
                        [5, 5], [6, 6], [7, 7], [0, 1], [1, 3], 
                        [0, 2], [2, 3], [3, 5], [2, 4], [5, 7], 
                        [4, 6]],
        "node_num": 8,
        "edge_num": 8
    },
    "pedestrian_graph_2": {
        "edge_indices": [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], 
                        [5, 5], [6, 6], [7, 7], [8, 8], [9, 9],
                        [10, 10], [11, 11], [12, 12], 
                        [0, 1], [0, 2], [1, 2], [1, 3], [3, 5], 
                        [2, 4], [4, 6], [1, 7], [2, 8], [7, 8], 
                        [7, 9], [8, 10], [9, 11], [10, 12]],
        "node_num": 13,
        "edge_num": 14
    },
    "combined_graph_2": {
        "edge_indices": [
                        [0, 0], [1, 1], [2, 2], [3, 3], [4, 4], 
                        [5, 5], [6, 6], [7, 7], # scene_graph_nodes
                        [8, 8], [9, 9], [10, 10], [11, 11], [12, 12], 
                        [13, 13], [14, 14], [15, 15], [16, 16], [17, 17],
                        [18, 18], [19, 19], [20, 20], # ped_graph_nodes
                        [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], 
                        [0, 6], [0, 7], # scene_graph_edges
                        [8, 9], [8, 10], [9, 10], [9, 11], [11, 13], 
                        [10, 12], [12, 14], [9, 15], [10, 16], [15, 16], 
                        [15, 17], [16, 18], [17, 19], [18, 20]], # ped_graph_edges
        "node_num": 21,
        "edge_num": 21
    }
}

"""
"scene_graph": {
        "edge_indices": [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4],
                        [5, 5], [6, 6], [7, 7], # scene_graph_nodes  
                        [0, 1], [0, 2],
                        [0, 3], [0, 4], [0, 5], [0, 6], [0, 7]], # scene_graph_edges
        "node_num": 8,
        "edge_num": 7
    },
"""

class NodeTypeEmbedding(nn.Module):
    """ Based on: https://github.com/jw9730/tokengt
    """
    def __init__(self, config, d_model):
        super(NodeTypeEmbedding, self).__init__()
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dim_hidden = d_model
        self.type_embedding = nn.Embedding(2, self.dim_hidden)

        graph_type = _get_graph_type_from_config(config)
        self.edge_indices = EDGE_NODE_VALS[graph_type]["edge_indices"]
        self.node_num = EDGE_NODE_VALS[graph_type]["node_num"]
        self.edge_num = EDGE_NODE_VALS[graph_type]["edge_num"]

    def forward(self, x):
        bsize = x.shape[0]
        edge12_indices = torch.tensor(self.edge_indices, dtype=torch.float, device=self.device)
        E = edge12_indices.shape[0]
        
        type_embedding = torch.zeros(bsize, E, self.dim_hidden, dtype=torch.float,
                                     device=self.device)  # [bsize, |E|, dim_hidden]

        for i in range(bsize):
            num_node_i = self.node_num
            num_edge12_i = x.shape[1]
            edge_type_index = torch.ones(num_edge12_i, dtype=torch.long, device=self.device)
            edge_type_index[:num_node_i] = torch.zeros(num_node_i, dtype=torch.long, device=self.device)
            type_emb_arr = self.type_embedding(edge_type_index)  # [num_edge12_i , dim_hidden]
            type_embedding[i, :num_edge12_i, :] = type_emb_arr
        return type_embedding
        # G = batch_like(G, G_type_value, skip_masking=False)
        
class LaplacianEmbedding(nn.Module):
    """ Based on: https://github.com/jw9730/tokengt
    """
    def __init__(self, config, d_model):
        super(LaplacianEmbedding, self).__init__()
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dim_hidden = d_model
        self.lap_node_id_dim = 42
        self.laplacian_encoder = nn.Linear(self.lap_node_id_dim, self.dim_hidden)

        graph_type = _get_graph_type_from_config(config)
        self.edge_indices = EDGE_NODE_VALS[graph_type]["edge_indices"]
        self.node_num = EDGE_NODE_VALS[graph_type]["node_num"]
        self.edge_num = EDGE_NODE_VALS[graph_type]["edge_num"]

    def forward(self, x):
        bsize = x.shape[0]
        edge12_indices = torch.tensor(self.edge_indices, dtype=torch.long, device=self.device)
        E = edge12_indices.shape[0] # all edges including nodes (|E|)
        
        sparse_edge_index = edge12_indices[self.node_num:, :]
        sparse_edge_index = sparse_edge_index.permute(1,0) # [|E|-|N|, 2]
        pe_list = []
        for i in range(bsize):
            pe_list.append(get_pe_2d(sparse_edge_index, edge12_indices, self.node_num, E,
                                     half_pos_enc_dim=self.lap_node_id_dim // 2,
                                     device=self.device))
        pe = torch.cat(pe_list)  # [bsize, |E|, lap_node_id_dim]
        pe = self.laplacian_encoder(pe)  # [bsize, |E|, dim_hidden]
        return pe
        #G_pe_value = G.values + pe  # [bsize, |E|, dim_hidden]
        #G = batch_like(G, G_pe_value, skip_masking=False)

def _get_graph_type_from_config(config):
    if hasattr(config, "graph_type"):
        if config.graph_type == "pedestrian_graph":
            graph_type = "pedestrian_graph_2"
        elif config.graph_type == "combined_graph":
            graph_type = "combined_graph_2"
        elif config.graph_type == "scene_graph":
            graph_type = "scene_graph_2"
    else:
        graph_type = "scene_graph"
    return graph_type