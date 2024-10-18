import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding
import numpy as np


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
        self.enc_embedding = DataEmbedding(configs.enc_in + 40, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        
        # Cross-Attention
        self.cross_attn_inner_dim = 10
        self.enc_box = DataEmbedding(4, self.cross_attn_inner_dim, configs.embed, configs.freq,
                                           configs.dropout)
        self.enc_abs_box = DataEmbedding(4, self.cross_attn_inner_dim, configs.embed, configs.freq,
                                           configs.dropout)
        self.enc_box_speed = DataEmbedding(2, self.cross_attn_inner_dim, configs.embed, configs.freq,
                                           configs.dropout)
        self.enc_speed = DataEmbedding(1, self.cross_attn_inner_dim, configs.embed, configs.freq,
                                           configs.dropout)
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
            self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                               configs.dropout)
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
            self.cross_attn_1 = AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), self.cross_attn_inner_dim, n_heads=1)
            self.cross_attn_2 = AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), self.cross_attn_inner_dim, n_heads=1)
            self.cross_attn_3 = AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), self.cross_attn_inner_dim, n_heads=1)
            self.cross_attn_4 = AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), self.cross_attn_inner_dim, n_heads=1)
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
        # enc_out = self.enc_embedding(x_enc, None)
        
        # Calculate cross-modality attention =====================
        box_values = self.enc_box(x_enc[:,:,0:4], x_mark=None)
        box_abs_values = self.enc_abs_box(x_enc[:,:,4:8], x_mark=None)
        box_speed_values = self.enc_box_speed(x_enc[:,:,8:10], x_mark=None)
        speed_values = self.enc_speed(x_enc[:,:,10].unsqueeze(2), x_mark=None) # nn.ReLU()(self.speed_emb_fc(trajectory_values[:,:,10].unsqueeze(2)))


        cross_attn_ctx_1, _ = self.cross_attn_1(box_values, speed_values, speed_values, attn_mask=None)
        cross_attn_ctx_2, _ = self.cross_attn_2(speed_values, box_values, box_values, attn_mask=None)
        cross_attn_ctx_3, _ = self.cross_attn_3(box_speed_values, box_abs_values, box_abs_values, attn_mask=None)
        cross_attn_ctx_4, _ = self.cross_attn_4(box_abs_values, box_speed_values, box_speed_values, attn_mask=None)
        
        enc_in = torch.cat([
            x_enc, cross_attn_ctx_1, cross_attn_ctx_2, cross_attn_ctx_3, cross_attn_ctx_4], dim=2)

        # Return to normal TF encoder
        enc_out = self.enc_embedding(enc_in, None)

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

"""
def collate_fn(data, max_len=None):
    Build mini-batch tensors from a list of (X, mask) tuples. Mask input. Create
    Args:
        data: len(batch_size) list of tuples (X, y).
            - X: torch tensor of shape (seq_length, feat_dim); variable seq_length.
            - y: torch tensor of shape (num_labels,) : class indices or numerical targets
                (for classification or regression, respectively). num_labels > 1 for multi-task models
        max_len: global fixed sequence length. Used for architectures requiring fixed length input,
            where the batch length cannot vary dynamically. Longer sequences are clipped, shorter are padded with 0s
    Returns:
        X: (batch_size, padded_length, feat_dim) torch tensor of masked features (input)
        targets: (batch_size, padded_length, feat_dim) torch tensor of unmasked features (output)
        target_masks: (batch_size, padded_length, feat_dim) boolean torch tensor
            0 indicates masked values to be predicted, 1 indicates unaffected/"active" feature values
        padding_masks: (batch_size, padded_length) boolean tensor, 1 means keep vector at this position, 0 means padding
"""