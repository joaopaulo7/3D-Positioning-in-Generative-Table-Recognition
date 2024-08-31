from transformers import VisionEncoderDecoderModel
import torch
from torch import nn, Tensor
import math

class PositionalEncoding(nn.Module):

    def __init__(self,  max_len: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.offset = 2
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.pos_enc = pe.squeeze(1)

    def forward(self, input_ids: torch.Tensor, past_key_values_length: int = 0):
        
        bsz, seq_len = input_ids.shape[:2]
        
        pos_start = past_key_values_length
        pos_end = past_key_values_length+seq_len
        
        positions = self.pos_enc[pos_start:pos_end].expand(bsz, -1, -1)
        
        return self.dropout(positions)



class PosDonutModel(VisionEncoderDecoderModel):

    def from_pretrained(*args, **kwargs):
        model = super(VisionEncoderDecoderModel, VisionEncoderDecoderModel).from_pretrained(*args, **kwargs)
        model.decoder.model.decoder.embed_positions = PositionalEncoding(kwargs["config"].decoder.max_position_encodings,
                                                                        kwargs["config"].decoder.d_model)
        return model
        
