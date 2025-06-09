import torch
from torch import nn
import math
from random import random
import torch
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.data import Dataset , DataLoader
from pathlib import Path
import matplotlib.pyplot as plt



'''Input Embedding --> the process of converting input text (words or subwords) into numerical vectors, capturing their semantic meaning'''

class InputEmbedding(nn.Module):
  def __init__(self, d_model:int, vocab_size:int):
    super().__init__()
    self.d_model = d_model
    self.vocab_size = vocab_size
    self.embedding = nn.Embedding(vocab_size, d_model)

  def forward(self, x):
    return self.embedding(x) * math.sqrt(self.d_model) # given in paper under Embedding and Softmax title

'''Positional Embedding --> a technique used to inject information about the position of words in a sequence into the model's architecture'''

class PositionalEncoding(nn.Module):

  def __init__(self , d_model,seq_length , dropout):
    super().__init__()
    self.d_model = d_model
    self.seq_length = seq_length
    self.dropout = nn.Dropout(p=dropout)

    # create a matrix of shape (seq_length , d_model)
    pe = torch.zeros(seq_length ,d_model)
    # create a vector which represent the position of the word in the sentence
    position = torch.arange(0,seq_length, dtype=torch.float).unsqueeze(1) # --> shape : [seq_length , 1]
    div_term = torch.exp(torch.arange(0 , d_model , 2).float()*(-math.log(10000)/d_model)) # --> shape : [d_model/2]
    # apply the sin to even pos and cos to odd pos
    pe[:,0::2] = torch.sin(position * div_term) # --> all the columns with rows from 0 with step of 2
    pe[:,1::2] = torch.cos(position * div_term)

    # now we add the batch dimension to apply to whole sentences
    pe = pe.unsqueeze(0) # --> shape: [1,seq_length , d_model]
    self.register_buffer('pe' , pe)

  def forward(self , x):
    # x.shape --> [batch_size, seq_length, d_model]
    x = x + (self.pe[: , :x.shape[1] , :]).requires_grad_(False)
    return self.dropout(x)

"""Layer Normalization --> normalization technique like batch normalization"""

class LayerNormalization(nn.Module):
  def __init__(self , epsilon:float = 10**-6 , d_model=512):
    super().__init__()
    self.eps = epsilon
    # nn.Parameter --> it is a special tensor that tells the model that it is a learnable parameter
    self.gamma = nn.Parameter(torch.ones(1,1,d_model))
    self.beta = nn.Parameter(torch.zeros(1,1,d_model))

  def forward(self , x):
    mean = x.mean(dim=-1 , keepdim=True)
    std = x.std(dim=-1 , keepdim=True)

    return self.gamma*(x-mean)/(std + self.eps) + self.beta

"""Feed Forward Layer --> This consists of two linear transformations(W1 , W2 , b1 , b2) with a ReLU activation(max-function) in between.
FFN(x) = max(0xW1 +b1)W2 +b2  
and the first layer in from d_model to d_ff and then the other one is from d_ff to d_model
"""
class FeedForwardLayer(nn.Module):
  def __init__(self , d_model , d_ff , dropout):
    super().__init__()
    self.feed_forward = nn.Sequential(
        nn.Linear(d_model , d_ff), # [batch , seq_length , d_model] --> [batch , seq_length , d_ff]
        nn.ReLU(),
        nn.Dropout(p=dropout),
        nn.Linear(d_ff , d_model),# [batch , seq_length , d_ff] --> [batch , seq_length , d_model]
    )

  def forward(self , x):
    return self.feed_forward(x)

'''Multi-Head-Attention --> a mechanism that enhances the original attention mechanism by running it multiple times in parallel, each with its own learnable parameters.
please watch "https://www.youtube.com/watch?v=bCz4OMemCcA" '''
class MultiheadAttention(nn.Module):
  def __init__(self , d_model , h , dropout):
    super().__init__()
    self.d_model = d_model
    self.num_heads = h
    self.d_k = d_model // h
    self.dropout = nn.Dropout(p=dropout)

    self.w_q = nn.Linear(d_model , d_model)
    self.w_k = nn.Linear(d_model , d_model)
    self.w_v = nn.Linear(d_model , d_model)

    self.w_o = nn.Linear(h*self.d_k, d_model) # h*d_k == d_model

    self.dropout = nn.Dropout(p=dropout)

  @staticmethod
  def attention(query , key , value ,mask , dropout):
    d_k = query.shape[-1]
    # remember --> key shape: [batch , num_heads , seq_length , d_k] , after transpose --> [batch , num_heads , d_k ,seq_length]
    attention_scores = (query @ key.transpose(-2 , -1))//math.sqrt(d_k) # --> [batch , num_heads , seq_length,seq_length]
    if mask is not None:
      attention_scores.masked_fill_(mask==0 , -1e9)
    attention_scores = attention_scores.softmax(dim=-1)

    if dropout is not None:
      attention_scores = dropout(attention_scores)

    return attention_scores @ value , attention_scores # --> shapes -> [batch , num_heads , seq_length , d_k] , [batch , num_heads , seq_length,seq_length]

  def forward(self, q,k,v, mask):
    # q.shape --> [batch , seq_length , d_model]
    query = self.w_q(q) # --> [batch , seq_length , d_model]
    key = self.w_k(k) # --> [batch , seq_length , d_model]
    value = self.w_v(v) # --> [batch , seq_length , d_model]

    # [batch , seq_length , d_model] --> [batch , seq_length , num_heads , d_k] --> [batch , num_heads , seq_length , d_k]
    query = query.view(query.shape[0] , query.shape[1] , self.num_heads , self.d_k).transpose(1,2)
    key = key.view(key.shape[0] , key.shape[1] , self.num_heads , self.d_k).transpose(1,2)
    value = value.view(value.shape[0] , value.shape[1] , self.num_heads , self.d_k).transpose(1,2)

    x , attention_scores = MultiheadAttention.attention(query , key , value , mask , self.dropout)

    # [batch , num_heads , seq_length , d_k] --> [batch , seq_length , num_heads , d_k] --> [batch , seq_length , d_model]
    x = x.transpose(1,2).contiguous().view(x.shape[0] , -1 , self.d_model)

    return self.w_o(x) # shape --> [batch , seq_length , d_model]

'''Residual Connection --> a technique used to stabilize the training of deep neural networks by adding a shortcut connection to the input of a layer'''
class ResidualConnection(nn.Module):
  def __init__(self , dropout):
    super().__init__()
    self.dropout = nn.Dropout(dropout)
    self.norm = LayerNormalization()

  def forward(self, x , sublayer):
    # sublayer is the prev layer
    return x + self.dropout(sublayer(self.norm(x)))

''' ************************    Encoder      ************************************************************
In a Transformer architecture, the encoder's key and value tensors are used by the decoder '''

# we expect the output from the encoder to be --> [batch , seq_length , d_model]
class EncoderBlock(nn.Module):
  def __init__(self ,self_attention_block:MultiheadAttention , feed_forward_block:FeedForwardLayer, dropout):
    super().__init__()
    self.self_attention_block = self_attention_block
    self.feed_forward_block = feed_forward_block
    self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

  def forward(self, x, mask):
    # first we do the self attention --> making the words intereact with each other in the same sentences
    x = self.residual_connections[0](x , lambda x: self.self_attention_block(x,x,x,mask))
    x = self.residual_connections[1](x , lambda y: self.feed_forward_block(y))

    return x # now this will go to the decoder as key and value pair

class Encoder(nn.Module):
  def __init__(self , layers:nn.Module):
    super().__init__()
    self.layers = layers
    self.norm = LayerNormalization()

  def forward(self , x ,mask):
    for layer in self.layers:
      x = layer(x , mask)

    return self.norm(x)

'''**********************   Decoder     ******************************************************8'''

class DecoderBlock(nn.Module):
  def __init__(self , self_attention_block:MultiheadAttention , cross_attention_block:MultiheadAttention , feed_forward_block:FeedForwardLayer, dropout):
    super().__init__()
    self.self_attention_block = self_attention_block
    self.cross_attention_block = cross_attention_block
    self.feed_forward_block = feed_forward_block
    self.residual_connections = [ResidualConnection(dropout) for _ in range(3)]

  def forward(self , x , encoder_output , encoder_mask,decoder_mask):
    x = self.residual_connections[0](x , lambda x: self.self_attention_block(x,x,x,decoder_mask))
    x = self.residual_connections[1](x , lambda x:self.cross_attention_block(x,encoder_output,encoder_output,encoder_mask)) # in this query will come from the masked multi-head-attention and the key and value will come from the encoder block output
    x = self.residual_connections[2](x , lambda y: self.feed_forward_block(y))

    return x

class Decoder(nn.Module):
  def __init__(self , layers:nn.Module):
    super().__init__()
    self.layers = layers
    self.norm = LayerNormalization()

  def forward(self , x , encoder_output , encoder_mask , decoder_mask):
    for layer in self.layers:
      x = layer(x , encoder_output , encoder_mask , decoder_mask)

    return self.norm(x)

# we expect the output form decoder to be --> [batch , seq_length , d_model]

# now we want to map these words into the vocabulary
class ProjectionLayer(nn.Module):
  def __init__(self , d_model , vocab_size):
    super().__init__()
    self.proj = nn.Linear(d_model , vocab_size)

  def forward(self , x):
    # x.shape --> [batch , seq_length , d_model]
    return torch.log_softmax(self.proj(x) ,dim=-1) # --> [batch , seq_length , vocab_size]

"""********************************     Transformer Block       *****************************************************  -->"""

class Transformer(nn.Module):
  def __init__(self , encoder:Encoder , decoder:Decoder , src_emb:InputEmbedding , trg_emb:InputEmbedding , srcpos:PositionalEncoding , trgpos:PositionalEncoding , projection_layer:ProjectionLayer):
    super().__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.src_emb = src_emb
    self.trg_emb = trg_emb
    self.srcpos = srcpos
    self.trgpos = trgpos
    self.projection_layer = projection_layer

  def encode(self , src , src_mask):
    # Convert input to long type
    src = src.long()
    src = self.srcpos(self.src_emb(src))
    return self.encoder(src , src_mask)

  def decode(self , trg , encoder_output , src_mask , trg_mask):
    # Convert input to long type
    trg = trg.long()
    trg = self.trgpos(self.trg_emb(trg))
    return self.decoder(trg , encoder_output , src_mask , trg_mask)

  def project(self , x):
    return self.projection_layer(x)

def build_transformer(src_vocab_size:int , trg_vocab_size:int , src_seq_len:int , trg_seq_len:int , d_model:int=512 , h:int=8 , dropout:float=0.1 , N:int=6,d_ff:int=2048)-> Transformer:
  # N --> no. of layers

  # creating the embeddings
  src_emb = InputEmbedding(d_model , src_vocab_size)
  trg_emb = InputEmbedding(d_model , trg_vocab_size)

  # positional encoding
  srcpos = PositionalEncoding(d_model , src_seq_len , dropout)
  trgpos = PositionalEncoding(d_model , trg_seq_len , dropout)

  # layers
  encoder_layer = EncoderBlock(MultiheadAttention(d_model , h, dropout) , FeedForwardLayer(d_model , d_ff , dropout) , dropout)
  decoder_layer = DecoderBlock(MultiheadAttention(d_model , h , dropout) , MultiheadAttention(d_model , h , dropout) , FeedForwardLayer(d_model , d_ff , dropout) , dropout)
  projection_layer = ProjectionLayer(d_model , trg_vocab_size)

  # build the encoder and decoder
  encoder = Encoder([encoder_layer for _ in range(N)])
  decoder = Decoder([decoder_layer for _ in range(N)])

  # finally the transformer
  transformer = Transformer(encoder , decoder , src_emb , trg_emb , srcpos , trgpos , projection_layer)

  # initialising the parameters with the xavier uniform
  for p in transformer.parameters():
    if p.dim() > 1:
      nn.init.xavier_uniform_(p)

  return transformer

if __name__ == "__main__":
    print("works")