import torch
from torch.utils.data import DataLoader
from datetime import datetime
import os
from model import CharDataset, CharTransformer

def load_model(vocab_size, model_path=None):
    """Initialize or load existing model"""
    model = CharTransformer(vocab_size)
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    return model

def train_model(text_file, model_path='quijote_transformer.pth', 
               epochs=10, batch_size=64, seq_length=50):
    """Train model on given text file"""
    # Load text
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Create dataset and model
    dataset = CharDataset(text, seq_length)
    model = load_model(len(dataset.char2idx), model_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(epochs):
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
        
        print(f'Epoch {epoch+1} completed - Avg Loss: {total_loss/batch_count:.4f}')
    
    # Save final model
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')
