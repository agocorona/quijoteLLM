import os
from training import train_model
from inference import inference_repl

# Main menu display function
def print_menu():
    print("\nQuijote LLM - Main Menu")
    print("=" * 30)
    print("1. Train new model")
    print("2. Continue training existing model")
    print("3. Run inference")
    print("4. Exit")
    print("=" * 30)

# Get valid text file path from user input
def get_text_file():
    while True:
        text_file = input("Enter path to text file: ")
        if os.path.exists(text_file):
            return text_file
        print(f"File not found: {text_file}")

# Get valid model file path from user input with optional default
def get_model_file(prompt="Enter model path (leave blank for default): "):
    while True:
        model_path = input(prompt).strip()
        if not model_path:
            return None  # Will use auto-generated name from training.py
        if os.path.exists(model_path):
            return model_path
        print(f"Model file not found: {model_path}")

# Main program loop handling user menu choices
def main():
    while True:
        print_menu()
        choice = input("Select option: ")
        
        if choice == '1':
            print("\nTraining New Model")
            print("=" * 30)
            text_file = get_text_file()
            model_path = input("Enter output model path (leave blank for default): ").strip()
            model_path = model_path or None  # Will use auto-generated name
            train_model(text_file, model_path)
            
        elif choice == '2':
            print("\nContinue Training")
            print("=" * 30)
            text_file = get_text_file()
            model_path = get_model_file()
            train_model(text_file, model_path)
            
        elif choice == '3':
            print("\nRun Inference")
            print("=" * 30)
            text_file = get_text_file()
            model_path = get_model_file()
            inference_repl(text_file, model_path)
            
        elif choice == '4':
            print("Exiting...")
            break
            
        else:
            print("Invalid choice. Please select 1-4.")

# Program entry point
if __name__ == '__main__':
    main()
