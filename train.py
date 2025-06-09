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
from dataset import BiLingualDataset
from Model import build_transformer


"""Building the tokenizer and data preparation"""

import torch
from torch.utils.data import Dataset, DataLoader

'''
# example of how i am using the following syntaxes
seq_length_for_example = 5
pad = torch.tensor([0])

print("for masking and learning how to use torch.triu ")
a = torch.triu(torch.ones(2,seq_length_for_example,seq_length_for_example) , diagonal=1)
print(a.shape)
print(a)
print(a==0)
print((a==0).int()) # thats how we are doing the masking in the decoder input


print()
print("learning how we are sending the masked encoder_input")

encoder_input = torch.cat([torch.randn([seq_length_for_example-3]) ,pad.repeat(3) ])

print(encoder_input)
print(encoder_input != pad)
print((encoder_input != pad).unsqueeze(0).unsqueeze(0).int())

print()
print("using caual_mask function created below")
print(casual_mask(seq_length_for_example).int().shape)'''


def get_all_sentences(ds , lang):
  "yielding all the sentences of a particular language"
  for item in ds:
    yield item['translation'][lang]


def get_or_build_tokenizer(config , da, lang):
  tokenizer_path = Path(config["tokenizer_file"].format(lang))
  if not tokenizer_path.exists():
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = WordLevelTrainer(special_tokens=["[UNK]","[PAD]","[SOS]","[EOS]"], min_frequency=2)
    tokenizer.train_from_iterator(get_all_sentences(da , lang) , trainer)

    tokenizer.save(str(tokenizer_path))
  else:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

  return tokenizer

DATASET_NAME = "opus100" 

def get_dataset(config):
  "all the work of building and converting the raw dataset into train and test dataset will be done here"
  ds_raw = load_dataset(DATASET_NAME , f'{config["lang_src"]}-{config["lang_tgt"]}',split="train", download_mode="force_redownload")

  # build tokenizer
  tokenizer_src = get_or_build_tokenizer(config , ds_raw , config["lang_src"])
  tokenizer_tgt = get_or_build_tokenizer(config , ds_raw , config["lang_tgt"])

  # keeping 90% for training
  train_ds_size = int(0.9 * len(ds_raw))
  test_ds_size = len(ds_raw) - train_ds_size

  train_ds , test_ds = torch.utils.data.random_split(ds_raw , [train_ds_size , test_ds_size])

  train_dataset = BiLingualDataset(train_ds , tokenizer_src , tokenizer_tgt , config["lang_src"] , config["lang_tgt"] , config["seq_length"])
  test_dataset = BiLingualDataset(test_ds , tokenizer_src , tokenizer_tgt , config["lang_src"] , config["lang_tgt"] , config["seq_length"])

  max_len_src = 0
  max_len_tgt = 0

  for item in ds_raw:
    src_ids = tokenizer_src.encode(item["translation"][config["lang_src"]]).ids
    tgt_ids = tokenizer_tgt.encode(item["translation"][config["lang_tgt"]]).ids

    max_len_src = max(max_len_src , len(src_ids))    # --> 417
    max_len_tgt = max(max_len_tgt , len(tgt_ids))    # --> 482

  print(f"max length of source text: {max_len_src}")
  print(f"max length of target text: {max_len_tgt}")

  return train_dataset , test_dataset , tokenizer_src , tokenizer_tgt , max_len_src , max_len_tgt


# change the language according to the about_dataset

config = {
    "lang_src": "en",
    "lang_tgt": "hi",
    "seq_length": 350,
    "tokenizer_file": "./tokenizer_{}.json",
}

train_ds, test_ds , tokenizer_src , tokenizer_tgt , max_len_src , max_len_tgt = get_dataset(config)

VOCAB_SRC_LEN = tokenizer_src.get_vocab_size()
VOCAB_TGT_LEN = tokenizer_tgt.get_vocab_size()
BATCH_SIZE = 32
NUM_EPOCHS = 20
LR = 10**-4
D_model = 512
D_ff = 2048
H = 8 # no of heads
N = 6 # no of layers
DROPOUT = 0.1

# making the dataloader for training and testing
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)



'''
for batch in train_loader:
  print(batch["encoder_input"].shape) # torch.Size([32, 350])
  print(batch["decoder_input"].shape) # torch.Size([32, 350])
  print(batch["label"].shape)         # torch.Size([32, 350])
  print(batch["encoder_mask"].shape)  # torch.Size([32, 1, 1, 350])
  print(batch["decoder_mask"].shape)  # torch.Size([32, 1, 350, 350])
  break
'''

"""********************   training the model    *****************************************************************"""
# building the model
model = build_transformer(VOCAB_SRC_LEN , VOCAB_TGT_LEN , config["seq_length"] , config["seq_length"] , D_model , H ,DROPOUT ,N , D_ff )
optimizer = torch.optim.Adam(model.parameters() , LR)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id("[PAD]") , label_smoothing=0.1)

losses = []

print("training the model")

for epoch in range(NUM_EPOCHS):
  print("in loop")
  model.train()
  for batch in train_loader:
    print("in batch loop")
    encoder_input = batch["encoder_input"].long()  # Convert to Long type
    decoder_input = batch["decoder_input"].long()  # Convert to Long type
    label = batch["label"].long()  # Convert to Long type
    encoder_mask = batch["encoder_mask"]
    decoder_mask = batch["decoder_mask"]

    # run the tensors through the transformer 
    encoder_output = model.encode(encoder_input , encoder_mask) # --> [batch , seq_length , d_model]
    decoder_output = model.decode(decoder_input , encoder_output , encoder_mask , decoder_mask) # [batch , seq_length , d_model]
    projection_output = model.projection_layer(decoder_output) # # --> [batch , seq_length , tgt_vocab_size]
    
    loss = criterion(projection_output.view(-1 , VOCAB_TGT_LEN) , label.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    print(f"{epoch}")

    if epoch%5==0:
      print(f"{epoch+1}/{NUM_EPOCHS} || loss : {loss.item()}")
    

plt.plot(losses)
plt.show()
