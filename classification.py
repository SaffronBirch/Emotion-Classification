import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
import torchtext.data
from torchtext.vocab import build_vocab_from_iterator
from pathlib import Path

# =========================
# Config
# =========================
base_path = Path(__file__).parent 
path = base_path / "Datasets/emotions.csv"

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"

MAX_VOCAB_SIZE = 20000
MAX_LEN = 100
EMBED_DIM = 128
HIDDEN_DIM = 128
NUM_LAYERS = 1
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# Load data
# =========================
if not os.path.exists(path):
    raise FileNotFoundError(f"Could not find {path}")

with open(path) as f:
    df = pd.read_csv(f)

if TEXT_COLUMN not in df.columns or LABEL_COLUMN not in df.columns:
    raise ValueError(f"Expected columns {TEXT_COLUMN!r} and {LABEL_COLUMN!r}. Found: {list(df.columns)}")

df = df[[TEXT_COLUMN, LABEL_COLUMN]].dropna()

texts = df[TEXT_COLUMN].astype(str).tolist()
raw_labels = df[LABEL_COLUMN].tolist()

# Map labels to contiguous IDs 0..C-1 (robust even if labels are strings)
unique_labels = sorted(set(raw_labels))
label2idx = {lab: i for i, lab in enumerate(unique_labels)}
idx2label = {i: lab for lab, i in label2idx.items()}
y = np.array([label2idx[lab] for lab in raw_labels], dtype=np.int64)

NUM_CLASSES = len(unique_labels)

print("Total samples:", len(texts))
print("Num classes:", NUM_CLASSES)
print("Labels (original -> id):", label2idx)


# =========================
# Train/val split
# =========================
X_train_texts, X_val_texts, y_train, y_val = train_test_split(
    texts,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,   # multiclass stratify
)

print("Train samples:", len(X_train_texts))
print("Val samples:", len(X_val_texts))


# =========================
# Tokenizer + vocab (same style as your example)
# ========================
tokenizer = torchtext.data.get_tokenizer("basic_english")

def yield_tokens(text_list):
    for t in text_list:
        yield tokenizer(str(t))

# Build vocab from TRAIN ONLY to avoid leakage
vocab = build_vocab_from_iterator(
    yield_tokens(X_train_texts),
    min_freq=5,
    specials=['<unk>', '<pad>', '<s>', '<eos>'])

UNK_IDX = vocab["<unk>"]
PAD_IDX = vocab["<pad>"]

vocab.set_default_index(UNK_IDX)

vocab_size = len(vocab)
print("Vocab size:", vocab_size)
print("PAD_IDX:", PAD_IDX, "UNK_IDX:", UNK_IDX)


def get_dataloaders(train_texts, train_labels, val_texts, val_labels,
                    vocab, tokenizer, max_length=100, batch_size=64):
    # --- TRAIN ---
    train_sequences = [
        torch.tensor(vocab.lookup_indices(tokenizer(str(text))), dtype=torch.int64)
        for text in train_texts
    ]
    train_padded = pad_sequence(
        train_sequences, batch_first=True, padding_value=PAD_IDX
    )[:, :max_length]
    train_targets = torch.tensor(train_labels, dtype=torch.int64)
    train_dataset = TensorDataset(train_padded, train_targets)

    # --- VAL ---
    val_sequences = [
        torch.tensor(vocab.lookup_indices(tokenizer(str(text))), dtype=torch.int64)
        for text in val_texts
    ]
    val_padded = pad_sequence(
        val_sequences, batch_first=True, padding_value=PAD_IDX
    )[:, :max_length]
    val_targets = torch.tensor(val_labels, dtype=torch.int64)
    val_dataset = TensorDataset(val_padded, val_targets)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)

    return train_loader, val_loader

train_loader, val_loader = get_dataloaders(
    X_train_texts, y_train,
    X_val_texts, y_val,
    vocab=vocab,
    tokenizer=tokenizer,
    max_length=MAX_LEN,
    batch_size=BATCH_SIZE
)

print("Batches (train):", len(train_loader))
print("Batches (val):", len(val_loader))


# =========================
# Model (BiLSTM multiclass)
# =========================
class EmotionBiLSTMModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        hidden_dim,
        num_layers,
        num_classes,
        pad_idx=0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(2 * hidden_dim, num_classes)

    def forward(self, input_ids):
        emb = self.embedding(input_ids)           # [B, L, E]
        _, (h_n, _) = self.lstm(emb)

        # final hidden states for forward/backward directions (last layer)
        h_forward = h_n[-2, :, :]                 # [B, H]
        h_backward = h_n[-1, :, :]                # [B, H]
        h_cat = torch.cat([h_forward, h_backward], dim=1)  # [B, 2H]

        x = self.dropout(h_cat)
        logits = self.fc(x)                       # [B, C]
        return logits

model = EmotionBiLSTMModel(
    vocab_size=vocab_size,
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    num_classes=NUM_CLASSES,
    pad_idx=PAD_IDX,
).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# =========================
# Train / eval loops
# =========================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for input_ids, labels in loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * input_ids.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    all_true = []
    all_pred = []

    for input_ids, labels in loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        logits = model(input_ids)
        loss = criterion(logits, labels)
        total_loss += loss.item() * input_ids.size(0)

        preds = torch.argmax(logits, dim=1)

        all_true.append(labels.cpu().numpy())
        all_pred.append(preds.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)

    y_true = np.concatenate(all_true)
    y_hat = np.concatenate(all_pred)

    f1_micro = f1_score(y_true, y_hat, average="micro", zero_division=0)
    f1_macro = f1_score(y_true, y_hat, average="macro", zero_division=0)

    return avg_loss, f1_micro, f1_macro, y_hat, y_true


best_val_f1 = 0.0

for epoch in range(1, EPOCHS + 1):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
    val_loss, f1_micro, f1_macro, y_hat, y_true = evaluate(model, val_loader, criterion, DEVICE)

    print(
        f"Epoch {epoch:02d} | "
        f"train loss: {train_loss:.4f} | "
        f"val loss: {val_loss:.4f} | "
        f"F1 micro: {f1_micro:.4f} | "
        f"F1 macro: {f1_macro:.4f}"
    )

print("\nClassification report:")
print(classification_report(
    y_true,
    y_hat,
    target_names=[str(idx2label[i]) for i in range(NUM_CLASSES)],
    zero_division=0
))




