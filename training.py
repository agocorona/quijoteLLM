import torch
from torch.utils.data import DataLoader, Dataset
from datetime import datetime
import os
import regex as re
from collections import Counter
from model import TransformerModel

# Byte Pair Encoding Tokenizer implementation
# Implements BPE algorithm for text tokenization
class BPETokenizer:
    def __init__(self, vocab_size=5000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = {}
        self.pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.chunk_size = 1024 * 1024  # Process 1MB at a time
        
    # Train the tokenizer on input text file
    # Args:
    #   text_file: Path to text file for training
    def train(self, text_file):
        print("\nStarting BPE training...")
        print("Pre-tokenizing text...", flush=True)
        
        # Process text in chunks to avoid memory issues
        vocab = Counter()
        with open(text_file, 'r', encoding='utf-8') as f:
            while True:
                chunk = f.read(self.chunk_size)
                if not chunk:
                    break
                words = self.pattern.findall(chunk)
                vocab.update(words)
        print(f"Found {len(words)} words with {len(vocab)} unique tokens", flush=True)
        
        # Initialize vocabulary with bytes
        print("Initializing base vocabulary...", flush=True)
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        
        # Perform BPE merges
        print(f"Starting BPE merges (target vocab size: {self.vocab_size})...", flush=True)
        merge_count = 0
        total_merges = self.vocab_size - 256
        
        while len(self.vocab) < self.vocab_size:
            pairs = self.get_stats(vocab)
            if not pairs:
                print("No more pairs to merge", flush=True)
                break
                
            best = max(pairs, key=pairs.get)
            vocab = self.merge_vocab(vocab, best)
            self.merges[best] = len(self.vocab)
            self.vocab[len(self.vocab)] = best[0] + best[1]
            
            merge_count += 1
            if merge_count % 50 == 0:
                print(f"Merge {merge_count}/{total_merges} - Vocab size: {len(self.vocab)}", flush=True)
        
        print(f"\nBPE training complete. Final vocab size: {len(self.vocab)}", flush=True)
        
    # Encode text into tokens using learned BPE vocabulary
    # Args:
    #   text: Input text to encode
    # Returns: List of token IDs
    def encode(self, text):
        import unicodedata
        # Normalize text to NFC form and ensure valid UTF-8
        text = unicodedata.normalize('NFC', text)
        
        words = self.pattern.findall(text)
        tokens = []
        for word in words:
            # Convert word to Unicode code points
            try:
                chars = [ord(c) for c in word]
                # Perform BPE merging on code points
                while len(chars) >= 2:
                    pairs = list(zip(chars[:-1], chars[1:]))
                    pair = min(pairs, key=lambda p: self.merges.get(p, float('inf')))
                    if pair not in self.merges:
                        break
                    idx = self.merges[pair]
                    chars = self.merge(chars, pair, idx)
                
                tokens.extend(chars)
            except Exception:
                tokens.append(32)  # Space character as fallback
        
        return tokens
    
    # Decode tokens back into text
    # Args:
    #   tokens: List of token IDs to decode
    # Returns: Decoded text string
    def decode(self, tokens):
        # Convert tokens to text using vocab
        text = []
        for token in tokens:
            try:
                token = int(token)
                if token in self.vocab:
                    # Get vocab entry and ensure it's a string
                    entry = self.vocab[token]
                    if isinstance(entry, bytes):
                        text.append(entry.decode('utf-8'))
                    elif isinstance(entry, str):
                        text.append(entry)
                    else:
                        text.append(' ')
                else:
                    text.append(' ')
            except (ValueError, TypeError):
                text.append(' ')
        
        # Join text and normalize Unicode
        decoded = ''.join(text)
        import unicodedata
        return unicodedata.normalize('NFC', decoded)
    
    def get_stats(self, vocab):
        pairs = Counter()
        for word, freq in vocab.items():
            # Convert word to bytes if it's a string
            if isinstance(word, str):
                word = tuple(ord(c) for c in word)
            symbols = tuple(word)
            for i in range(len(symbols)-1):
                # Ensure we're working with integers
                pair = (int(symbols[i]), int(symbols[i+1]))
                pairs[pair] += freq
        return pairs
    
    def merge_vocab(self, vocab, pair):
        new_vocab = {}
        # Convert pair to tuple of valid bytes (0-255)
        if isinstance(pair, str):
            pair = tuple(ord(c) % 256 for c in pair)
        elif not isinstance(pair, tuple):
            pair = tuple(pair)
        pair = tuple(b % 256 for b in pair)
            
        try:
            # Create byte string from valid bytes
            byte_pair = bytes(pair)
            pattern = re.escape(byte_pair.decode('latin1', errors='replace'))
            p = re.compile(pattern)
            
            for word in vocab:
                # Convert word to valid bytes
                if isinstance(word, str):
                    word = tuple(ord(c) % 256 for c in word)
                else:
                    word = tuple(b % 256 for b in word)
                    
                # Convert to string with error handling
            try:
                word_str = bytes(word).decode('utf-8', errors='strict')
                # Perform replacement
                w_out = p.sub(byte_pair.decode('utf-8', errors='strict'), word_str)
                # Convert back to valid bytes
                new_word = tuple(w_out.encode('utf-8', errors='strict'))
                new_vocab[new_word] = vocab[word]
            except UnicodeError:
                # Skip invalid UTF-8 sequences
                new_vocab[word] = vocab[word]
        except Exception as e:
            print(f"Error processing pair {pair}: {str(e)}")
            return vocab
            
        return new_vocab
    
    def merge(self, word, pair, idx):
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word)-1 and (word[i], word[i+1]) == pair:
                new_word.append(idx)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        return tuple(new_word)

# PyTorch Dataset for text training data
# Handles tokenization and sequence generation
class TextDataset(Dataset):
    def __init__(self, text_file, tokenizer, seq_length=128):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.text_file = text_file
        
        # Calculate total sequences by reading file size
        self.file_size = os.path.getsize(text_file)
        self.total_sequences = self.file_size // (seq_length * 4)  # Estimate
        
    def __len__(self):
        return self.total_sequences
    
    def __getitem__(self, idx):
        # Calculate file position
        start_pos = idx * self.seq_length * 4  # Estimate based on avg token size
        with open(self.text_file, 'rb') as f:
            f.seek(start_pos)
            # Read enough text for sequence
            text = f.read(self.seq_length * 10)  # Read extra to ensure enough tokens
            text = text.decode('utf-8', errors='replace')
            tokens = self.tokenizer.encode(text)
            if len(tokens) < self.seq_length + 1:
                # If not enough tokens, pad with zeros
                tokens += [0] * (self.seq_length + 1 - len(tokens))
            x = torch.tensor(tokens[:self.seq_length], dtype=torch.long)
            y = torch.tensor(tokens[1:self.seq_length+1], dtype=torch.long)
            return x, y

def load_model(vocab_size, model_path=None):
    """Initialize or load existing model"""
    model = TransformerModel(
        vocab_size=vocab_size,
        n_embd=512,
        n_head=8,
        n_layer=6,
        block_size=128
    )
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    return model

# Main training function for the transformer model
# Args:
#   text_file: Path to text file for training
#   model_path: Path to save trained model
#   epochs: Number of training epochs
#   batch_size: Training batch size
#   seq_length: Sequence length for training
def train_model(text_file, model_path=None, 
               epochs=10, batch_size=64, seq_length=128):
    """Train model on given text file"""
    # Set model path based on input file name if not provided
    if model_path is None:
        base_name = os.path.splitext(os.path.basename(text_file))[0]
        model_path = os.path.join(os.path.dirname(text_file), f"{base_name}_transformer.pth")
    
    # Initialize tokenizer
    tokenizer = BPETokenizer(vocab_size=5000)
    tokenizer.train(text_file)
    
    # Create dataset and model
    dataset = TextDataset(text_file, tokenizer, seq_length)
    model = load_model(len(tokenizer.vocab), model_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(epochs):
        epoch_start = datetime.now()
        total_loss = 0
        batch_count = 0
        
        for i, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output.view(-1, len(tokenizer.vocab)), y.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            # Print progress every 50 batches
            if i % 50 == 0:
                elapsed = datetime.now() - epoch_start
                avg_loss = total_loss / batch_count
                print(f'Epoch {epoch+1} [{i}/{len(dataloader)}] - Loss: {avg_loss:.4f} - Elapsed: {elapsed}', flush=True)
        
        print(f'Epoch {epoch+1} completed - Avg Loss: {total_loss/batch_count:.4f}', flush=True)
    
    # Save final model and tokenizer
    if not model_path:
        base_name = os.path.splitext(os.path.basename(text_file))[0]
        model_path = os.path.join(os.path.dirname(text_file), f"{base_name}_transformer.pth")
    torch.save(model.state_dict(), model_path)
    torch.save(tokenizer, f"{model_path}.tokenizer")
    print(f'Model saved to {model_path}')
    print(f'Tokenizer saved to {model_path}.tokenizer')
