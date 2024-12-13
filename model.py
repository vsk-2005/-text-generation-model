import torch
import torch.nn as nn
import string
import random
import os

class TextRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(TextRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden):
        x = self.embedding(x)
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return out, hidden
    
    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)

class TextGenerator:
    def __init__(self, file_path=None, text=None):
        if file_path is None and text is None:
            raise ValueError("Either file_path or text must be provided")
            
        # Load text from file if provided
        if file_path is not None:
            self.training_text = self.load_text(file_path)
        else:
            self.training_text = text
            
        # Define character set
        self.chars = sorted(list(set(self.training_text)))
        self.char_to_idx = {char: idx for idx, char in enumerate(self.chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        
        # Model parameters
        self.hidden_size = 256  # Increased for better handling of larger texts
        self.num_layers = 2
        
        # Initialize model
        self.model = TextRNN(
            input_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=self.vocab_size
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    @staticmethod
    def load_text(file_path):
        """Load and preprocess text from a file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return text
    
    def save_model(self, path):
        """Save the model to a file"""
        checkpoint = {
            'model_state': self.model.state_dict(),
            'char_to_idx': self.char_to_idx,
            'idx_to_char': self.idx_to_char,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers
        }
        torch.save(checkpoint, path)
        
    def load_model(self, path):
        """Load a saved model"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
            
        checkpoint = torch.load(path, map_location=self.device)
        
        # Recreate the model with saved parameters
        self.hidden_size = checkpoint['hidden_size']
        self.num_layers = checkpoint['num_layers']
        self.char_to_idx = checkpoint['char_to_idx']
        self.idx_to_char = checkpoint['idx_to_char']
        self.vocab_size = len(self.char_to_idx)
        
        self.model = TextRNN(
            self.vocab_size,
            self.hidden_size,
            self.num_layers,
            self.vocab_size
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state'])
    
    def prepare_sequence(self, seq):
        """Convert string sequence to tensor of indices"""
        return torch.tensor([self.char_to_idx[c] for c in seq], dtype=torch.long).to(self.device)
    
    def train(self, seq_length=100, num_epochs=100, batch_size=1, print_every=10):
        """Train the model on the loaded text"""
        print(f"Training on {self.device}")
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Text length: {len(self.training_text)} characters")
        
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            hidden = self.model.init_hidden(batch_size).to(self.device)
            
            # Create training sequences
            for i in range(0, len(self.training_text) - seq_length - 1, seq_length):
                # Get input and target sequences
                input_seq = self.training_text[i:i + seq_length]
                target_seq = self.training_text[i + 1:i + seq_length + 1]
                
                # Convert to tensors
                input_tensor = self.prepare_sequence(input_seq).unsqueeze(0)
                target_tensor = self.prepare_sequence(target_seq).unsqueeze(0)
                
                # Forward pass
                output, hidden = self.model(input_tensor, hidden)
                hidden = hidden.detach()
                
                # Calculate loss
                loss = self.criterion(output.view(-1, self.vocab_size), target_tensor.view(-1))
                total_loss += loss.item()
                
                # Backward pass and optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)  # Add gradient clipping
                self.optimizer.step()
            
            if (epoch + 1) % print_every == 0:
                avg_loss = total_loss / (len(self.training_text) // seq_length)
                print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')
    
    def generate(self, prompt, max_length=200, temperature=0.8):
        """Generate text given a prompt"""
        self.model.eval()
        hidden = self.model.init_hidden(1).to(self.device)
        input_sequence = self.prepare_sequence(prompt).unsqueeze(0)
        output_text = prompt
        
        with torch.no_grad():
            for _ in range(max_length):
                # Get prediction
                output, hidden = self.model(input_sequence, hidden)
                
                # Apply temperature to output probabilities
                output = output[:, -1, :] / temperature
                probs = torch.softmax(output, dim=-1)
                
                # Sample next character
                next_char_idx = torch.multinomial(probs, 1).item()
                next_char = self.idx_to_char[next_char_idx]
                
                # Add to output text
                output_text += next_char
                
                # Update input sequence
                input_sequence = torch.tensor([[next_char_idx]], dtype=torch.long).to(self.device)
        
        return output_text

# Example usage
def main():
    # File paths
    input_file = "C:/Users/venkat sai/Desktop/mini/archive (1)/1of2/wiki_01.txt"  # Your text file
    model_save_path = "C:/Users/venkat sai/Desktop/mini/text_generator.pth"  # Path to save the model
    
    # Initialize and train the model
    generator = TextGenerator(file_path=input_file)
    
    # Train the model
    generator.train(num_epochs=50)
    
    # Save the model
    generator.save_model(model_save_path)
    
    # Assuming generator is already defined and imported above

    # Collect a single prompt from the user
    prompt = input("Enter a prompt: ")

    # Generate text for the user-provided prompt
    generated_text = generator.generate(prompt, max_length=200)
    print(f"\nPrompt: {prompt}")
    print(f"Generated text: {generated_text}")
    print("-" * 50)


if __name__ == "__main__":
    main()