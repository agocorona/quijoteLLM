import os
import torch
from training import BPETokenizer
from model import TransformerModel

# Initialize or load a transformer model
# Args:
#   vocab_size: Size of vocabulary for token embeddings
#   model_path: Optional path to load existing model weights
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

# Generate text using the model with given starting text
# Args:
#   model: Transformer model instance
#   tokenizer: Tokenizer for text encoding/decoding
#   start_text: Initial text to start generation from
#   length: Number of tokens to generate
#   temperature: Sampling temperature (higher = more random)
def generate_text(model, tokenizer, start_text, length=100, temperature=0.8):
    """Generate text from a given starting text"""
    model.eval()
    tokens = tokenizer.encode(start_text)
    x = torch.tensor(tokens).unsqueeze(0)
    
    for _ in range(length):
        with torch.no_grad():
            output = model(x)
            probs = torch.softmax(output[:, -1] / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            tokens.append(next_token)
            x = torch.tensor(tokens[-128:]).unsqueeze(0)
            
    return tokenizer.decode(tokens)

# Run interactive inference REPL (Read-Eval-Print Loop)
# Args:
#   model_path: Path to saved model weights
def inference_repl(text_file, model_path=None):
    """Run interactive inference REPL"""
    # Load model and tokenizer
    model = load_model(5000)  # Default vocab size
    if not model_path:
        model_path = os.path.splitext(text_file)[0] + "_transformer.pth"
    model.load_state_dict(torch.load(model_path))
    tokenizer = torch.load(f"{model_path}.tokenizer")
    
    print("\nInteractive Inference Mode")
    print("Type 'quit' to exit")
    print("=" * 50)
    
    while True:
        start_text = input("\nEnter starting text: ")
        if start_text.lower() == 'quit':
            break
            
        length = int(input("Enter length (50-500): "))
        temperature = float(input("Enter temperature (0.1-1.5): "))
        
        generated = generate_text(model, tokenizer, start_text, length, temperature)
        print("\nGenerated text:")
        print(generated)
        print("=" * 50)
