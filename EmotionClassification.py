import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
import torchtext.data
from torchtext.vocab import build_vocab_from_iterator
from pathlib import Path



# Load the dataset
base_path = Path(__file__).parent 
path = base_path / "Dataset/isear.csv"


df = pd.read_csv(path, header=None)

df.drop(df[df[1] == '[ No response.]'].index, inplace = True)
print(df.head())

# Tokenization

tokenizer = torchtext.data.get_tokenizer('basic_english')


