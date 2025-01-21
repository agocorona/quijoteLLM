import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class CharDataset(Dataset):
    def __init__(self, text, seq_length=50):
        # Use only first 25% of text
        text = text[:len(text)//4]
        chars = sorted(list(set(text)))
        self.char2idx = {ch: i for i, ch in enumerate(chars)}
        self.idx2char = {i: ch for i, ch in enumerate(chars)}
        self.data = [self.char2idx[ch] for ch in text]
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.data) - self.seq_length
        
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_length]
        y = self.data[idx+1:idx+self.seq_length+1]
        return torch.tensor(x), torch.tensor(y)

class CharTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 50, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        seq_len = x.size(1)
        x = self.embedding(x) + self.pos_encoder[:, :seq_len]
        x = self.transformer(x)
        return self.fc_out(x)
