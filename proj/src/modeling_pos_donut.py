# part of the __init__() code sourced from now deleted pytorch tutorial available in:
#   https://github.com/pytorch/tutorials/blob/57bad606c011a83607a16a5bb89a652b90d7a307/beginner_source/transformer_tutorial.py#L114
# under the following license:
#BSD 3-Clause License
#Copyright (c) 2017-2022, Pytorch contributors
#All rights reserved.
#
#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#
#* Redistributions of source code must retain the above copyright notice, this
#  list of conditions and the following disclaimer.
#
#* Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#* Neither the name of the copyright holder nor the names of its
#  contributors may be used to endorse or promote products derived from
#  this software without specific prior written permission.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
        model = VisionEncoderDecoderModel.from_pretrained(*args, **kwargs)
        
        
        model.decoder.model.decoder.embed_positions = PositionalEncoding(kwargs["config"].decoder.max_position_encodings,
                                                                        kwargs["config"].decoder.d_model)
        return model
        
