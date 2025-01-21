import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import string
import os
from datetime import datetime

# Preprocessing
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

# Transformer Model
class CharTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 50, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, batch_first=True)  # Added batch_first
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        seq_len = x.size(1)
        x = self.embedding(x) + self.pos_encoder[:, :seq_len]
        x = self.transformer(x)
        return self.fc_out(x)

# Training
def train_model():
    # Load text
    with open('quijoteLLM/quijote.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Create dataset
    dataset = CharDataset(text)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Initialize model
    model = CharTransformer(vocab_size=len(dataset.char2idx))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Checkpointing
    checkpoint_path = '/home/quijoteLLM/checkpoint.pth'
    start_epoch = 0
    
    # Load checkpoint if exists
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
    
    # Training loop
    for epoch in range(start_epoch, 10):
        epoch_start = datetime.now()
        total_loss = 0
        batch_count = 0
        
        for i, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output.view(-1, len(dataset.char2idx)), y.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            # Print progress every 10 batches
            if i % 10 == 0:
                elapsed = datetime.now() - epoch_start
                avg_loss = total_loss / batch_count
                print(f'Epoch {epoch+1} [{i}/{len(dataloader)}] - Loss: {avg_loss:.4f} - Elapsed: {elapsed}')
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss / batch_count,
        }, checkpoint_path)
        
        print(f'Epoch {epoch+1} completed - Avg Loss: {total_loss/batch_count:.4f}')
    
    # Save final model
    torch.save(model.state_dict(), 'quijote_transformer.pth')

def generate_text(model, dataset, start_text, length=100, temperature=0.8):
    model.eval()
    chars = [dataset.char2idx[ch] for ch in start_text]
    x = torch.tensor(chars).unsqueeze(0)
    
    for _ in range(length):
        with torch.no_grad():
            output = model(x)
            probs = F.softmax(output[:, -1] / temperature, dim=-1)
            next_char = torch.multinomial(probs, 1).item()
            chars.append(next_char)
            x = torch.tensor(chars[-50:]).unsqueeze(0)
            
    return ''.join([dataset.idx2char[i] for i in chars])

def infer_model(start_text, length=200, temperature=0.8):
    """Generate text from a given starting text"""
    # Load text and dataset
    with open('quijoteLLM/quijote.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    dataset = CharDataset(text)
    
    # Load model
    model = CharTransformer(vocab_size=len(dataset.char2idx))
    model.load_state_dict(torch.load('quijote_transformer.pth'))
    
    # Generate text
    generated = generate_text(model, dataset, start_text, length, temperature)
    return generated

def run_inference_examples():
    """Run multiple inference examples from the training text"""
    # Load text
    with open('quijoteLLM/quijote.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Select random starting points from the text
    example_starts = [
        "En un lugar de la Mancha",
        "Don Quijote de la Mancha",
        "La razón de la sinrazón",
        "En esto descubrieron treinta o cuarenta molinos",
        "Dichosa edad y siglos dichosos"
    ]
    
    print("\nRunning Inference Examples:")
    print("=" * 50)
    for i, start_text in enumerate(example_starts):
        print(f"\nExample {i+1}:")
        print(f"Starting text: {start_text}")
        generated = infer_model(start_text)
        print("\nGenerated continuation:")
        print(generated)
        print("-" * 50)

if __name__ == '__main__':
    train_model()
    run_inference_examples()
