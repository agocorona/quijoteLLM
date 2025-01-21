import torch
import torch.nn.functional as F
from model import CharDataset, CharTransformer

def load_model_and_dataset(text_file, model_path):
    """Load model and create dataset from text file"""
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    dataset = CharDataset(text)
    model = CharTransformer(vocab_size=len(dataset.char2idx))
    model.load_state_dict(torch.load(model_path))
    return model, dataset

def generate_text(model, dataset, start_text, length=100, temperature=0.8):
    """Generate text from a given starting text"""
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

def inference_repl(text_file, model_path):
    """Run interactive inference REPL"""
    model, dataset = load_model_and_dataset(text_file, model_path)
    
    print("\nInteractive Inference Mode")
    print("Type 'quit' to exit")
    print("=" * 50)
    
    while True:
        start_text = input("\nEnter starting text: ")
        if start_text.lower() == 'quit':
            break
            
        length = int(input("Enter length (50-500): "))
        temperature = float(input("Enter temperature (0.1-1.5): "))
        
        generated = generate_text(model, dataset, start_text, length, temperature)
        print("\nGenerated text:")
        print(generated)
        print("=" * 50)
