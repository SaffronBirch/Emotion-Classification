import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim
from lightning.pytorch import LightningModule
from torchmetrics import Accuracy
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score, classification_report
import torchtext.data
from torchtext.vocab import build_vocab_from_iterator
from pathlib import Path
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import CSVLogger
from torchmetrics import F1Score
from torch.nn.utils.rnn import pack_padded_sequence


# Load the dataset
base_path = Path.cwd() 
path = base_path / "Dataset/isear.csv"

df = pd.read_csv(path, header=None, names=["emotion", "text"])
#df = pd.DataFrame(df, columns=['emotion', 'text'])
df = df[['emotion', 'text']].dropna()

df.drop(df[df['text'] == '[ No response.]'].index, inplace = True)

# Map each unique emotion string -> a stable integer id
labels = sorted(df["emotion"].unique())
label2id = {lab: i for i, lab in enumerate(labels)}
id2label = {i: lab for lab, i in label2id.items()}

df["emotion_id"] = df["emotion"].map(label2id).astype(int)
num_classes = len(labels)

print("num_classes:", num_classes)
print("label2id:", label2id)

# Tokenization
tokenizer = torchtext.data.get_tokenizer('basic_english')

def iterate_tokens(df):
    for text in df['text']:
        yield tokenizer(text)

# Vocab
vocab = build_vocab_from_iterator(
    iterate_tokens(df),
    min_freq=5,
    specials=['<pad>', '<unk>', '<s>', '<eos>'])

vocab.set_default_index(vocab["<unk>"])
pad_idx = vocab["<pad>"]

# Padding
sequences = [
    torch.tensor(vocab.lookup_indices(tokenizer(text), ), dtype=torch.int64)\
    for text in df['text']
]

lengths = torch.tensor([min(len(seq), 250) for seq in sequences], dtype=torch.int64)
padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=pad_idx)[:, :250]

emotions = torch.tensor(df['emotion_id'].to_numpy(), dtype=torch.int64)

dataset = TensorDataset(padded_sequences, lengths, emotions)
(train_dataset, val_dataset) = random_split(dataset, (0.8, 0.2))

batch_size = 32
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)

# LSTM Classification
class emotionLSTM(LightningModule):
    def __init__(self, vocab_size, dim_emb, dim_state, lr):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, dim_emb, padding_idx=pad_idx)

        self.lstm = nn.LSTM(input_size=dim_emb,
                            hidden_size=dim_state,
                            num_layers=1,
                            batch_first=True)
        
        self.output = nn.Linear(dim_state, num_classes)

        self.accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.f1_macro = F1Score(task='multiclass', num_classes=num_classes, average="macro")
        self.f1_micro = F1Score(task='multiclass', num_classes=num_classes, average="micro")

        self.lr = lr

    def forward(self, seq_batch, lengths):
        emb = self.embedding(seq_batch)
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (state, _) = self.lstm(packed)
        output = self.output(state[-1])
        return output
    
    def loss(self, outputs, targets):
        return nn.functional.cross_entropy(outputs, targets)
    
    def training_step(self, batch, batch_index):
        inputs, lengths, targets = batch
        outputs = self.forward(inputs, lengths)
        
        loss = self.loss(outputs, targets)

        preds = torch.argmax(outputs, dim=1)
        self.accuracy(preds, targets)

        self.log('acc', self.accuracy, prog_bar=True)
        self.log('loss', loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def validation_step(self, batch, batch_index):
        inputs, lengths, targets = batch
        outputs = self.forward(inputs, lengths)

        preds = torch.argmax(outputs, dim=1)

        self.accuracy(preds, targets)
        self.f1_macro(preds, targets)
        self.f1_micro(preds, targets)

        self.log('val_acc', self.accuracy, prog_bar=True)
        self.log("val_f1_macro", self.f1_macro, prog_bar=True)
        self.log("val_f1_micro", self.f1_micro, prog_bar=True)
    
    
# Training
logger = CSVLogger('./lightning_logs/', 'emotion_lstm')

trainer = Trainer(max_epochs=20, logger=logger)

model = emotionLSTM(
    vocab_size=len(vocab),
    dim_emb=32,
    dim_state=64,
    lr=1e-3)

trainer.fit(model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader)

# Classification Report
model.eval()

all_preds = []
all_targets = []

with torch.no_grad():
    for inputs, lengths, targets in val_dataloader:
        outputs = model(inputs, lengths)

        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

print(classification_report(
    all_targets,
    all_preds,
    target_names=[id2label[i] for i in range(num_classes)],
    digits=4
))


micro_f1 = f1_score(all_targets, all_preds, average="micro")
macro_f1 = f1_score(all_targets, all_preds, average="macro")
weighted_f1 = f1_score(all_targets, all_preds, average="weighted")

print(f"micro F1:    {micro_f1:.4f}")
print(f"macro F1:    {macro_f1:.4f}")
print(f"weighted F1: {weighted_f1:.4f}")

metrics = pd.read_csv("./lightning_logs/emotion_lstm/version_1/metrics.csv")
f1_curve = metrics["val_f1_macro"].dropna().reset_index(drop=True)
print(f1_curve)

    

