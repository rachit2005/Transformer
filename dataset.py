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


'''Building the dataset'''
def casual_mask(size):
    # Returns the upper triangular part of a matrix (2-D tensor) or batch of matrices input, the other elements of the result tensor out are set to 0.
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0

class BiLingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_length):
        super().__init__()
        self.seq_length = seq_length

        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt

        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.ds = ds

        self.sos = torch.tensor([tokenizer_src.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos = torch.tensor([tokenizer_src.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad = torch.tensor([tokenizer_src.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        src_target_pair = self.ds[index]

        input = src_target_pair['translation'][self.src_lang]
        label = src_target_pair['translation'][self.tgt_lang]

        enc_input_tokens = self.tokenizer_src.encode(input).ids
        dec_input_tokens = self.tokenizer_tgt.encode(label).ids

        enc_num_padding = self.seq_length - len(enc_input_tokens) - 2  # we will also add the sos and eos token that's why we subtract it by 2 too!.
        dec_num_padding = self.seq_length - len(dec_input_tokens) - 1  # in decoder input we add only sos token and in label eos label thats it!.

        if enc_num_padding < 0 or dec_num_padding < 0:
            raise ValueError("sentence is too long !!!! ..")

        # Convert all tensors to Long type
        encoder_input = torch.cat([
            self.sos,
            torch.tensor(enc_input_tokens, dtype=torch.int64),
            self.eos,
            torch.tensor([self.pad] * enc_num_padding, dtype=torch.int64)
        ])

        decoder_input = torch.cat([
            self.sos,
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            torch.tensor([self.pad] * dec_num_padding, dtype=torch.int64)
        ])

        label = torch.cat([
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            self.eos,
            torch.tensor([self.pad] * dec_num_padding, dtype=torch.int64)
        ])

        assert encoder_input.size(0) == self.seq_length
        assert decoder_input.size(0) == self.seq_length
        assert label.size(0) == self.seq_length

        return {
            "encoder_input": encoder_input,  # [seq_length]
            "decoder_input": decoder_input,  # [seq_length]
            "label": label,  # [seq_length]
            "encoder_mask": (encoder_input != self.pad).unsqueeze(0).unsqueeze(0).int(),  # [1,1,seq_length]
            "decoder_mask": (decoder_input != self.pad).unsqueeze(0).int() & casual_mask(decoder_input.size(0)).int(),  # [1,seq_length] & [1,seq_length,seq_length]
            "src_text": input,
            "tgt_text": src_target_pair['translation'][self.tgt_lang],  # Use the original target text
        }
    

if __name__ == "__main__":
    print("works")