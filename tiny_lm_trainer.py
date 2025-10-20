"""
Tiny Language Model Trainer (~1M parameters)
This script trains a small transformer model from scratch on the WikiText-2 dataset.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer
import math
from tqdm import tqdm
import os

# Configuration
class Config:
    # Model parameters
    vocab_size = 50257  # GPT-2 tokenizer vocab size
    max_seq_length = 128
    embed_dim = 128
    num_layers = 4
    num_heads = 4
    ff_dim = 512
    dropout = 0.1
    
    # Training parameters
    batch_size = 16
    learning_rate = 3e-4
    num_epochs = 3
    save_every = 1000
    eval_every = 500
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Paths
    model_save_path = "tiny_lm_model.pt"

config = Config()

# Simple Transformer Model
class TinyLanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.position_embedding = nn.Embedding(config.max_seq_length, config.embed_dim)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        
        # Output layer
        self.ln_f = nn.LayerNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_length = input_ids.shape
        
        # Get embeddings
        positions = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_length)
        
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(positions)
        x = token_embeds + position_embeds
        
        # Create causal mask
        causal_mask = torch.triu(torch.ones(seq_length, seq_length, device=input_ids.device), diagonal=1).bool()
        
        # Pass through transformer
        x = self.transformer(x, mask=causal_mask, src_key_padding_mask=attention_mask)
        
        # Output layer
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits

# Dataset class
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        print("Tokenizing dataset...")
        for text in tqdm(texts):
            if text.strip():  # Skip empty texts
                tokens = tokenizer.encode(text, add_special_tokens=True, max_length=max_length, truncation=True)
                if len(tokens) > 1:  # Need at least 2 tokens for input/target
                    self.examples.append(tokens)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        tokens = self.examples[idx]
        # Pad if needed
        if len(tokens) < self.max_length:
            tokens = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        return torch.tensor(tokens[:self.max_length], dtype=torch.long)

def load_data(tokenizer, config):
    """Load WikiText-103 dataset (much larger and better)"""
    print("Downloading dataset (this may take a few minutes)...")
    dataset = load_dataset("wikitext", "wikitext-103-v1")
    
    print("Preparing training data...")
    train_dataset = TextDataset(dataset["train"]["text"], tokenizer, config.max_seq_length)
    
    print("Preparing validation data...")
    val_dataset = TextDataset(dataset["validation"]["text"], tokenizer, config.max_seq_length)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    
    return train_loader, val_loader

def calculate_perplexity(model, dataloader, device):
    """Calculate perplexity on a dataset"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            
            # Input is all tokens except last, target is all tokens except first
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            
            logits = model(inputs)
            
            # Calculate loss
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, config.vocab_size),
                targets.reshape(-1),
                ignore_index=tokenizer.pad_token_id
            )
            
            # Count non-padding tokens
            non_pad_tokens = (targets != tokenizer.pad_token_id).sum().item()
            total_loss += loss.item() * non_pad_tokens
            total_tokens += non_pad_tokens
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity

def train(model, train_loader, val_loader, config, tokenizer):
    """Train the model"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    model.to(config.device)
    global_step = 0
    
    print(f"\nTraining on {config.device}...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            batch = batch.to(config.device)
            
            # Prepare input and target
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            
            # Forward pass
            logits = model(inputs)
            loss = criterion(logits.reshape(-1, config.vocab_size), targets.reshape(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # Evaluate periodically
            if global_step % config.eval_every == 0:
                perplexity = calculate_perplexity(model, val_loader, config.device)
                print(f"\nStep {global_step} - Validation Perplexity: {perplexity:.2f}")
                model.train()
            
            # Save checkpoint
            if global_step % config.save_every == 0:
                torch.save(model.state_dict(), config.model_save_path)
                print(f"\nModel saved to {config.model_save_path}")
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
    
    # Final save
    torch.save(model.state_dict(), config.model_save_path)
    print(f"\nTraining complete! Model saved to {config.model_save_path}")

def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.8, device="cpu"):
    """Generate text from a prompt"""
    model.eval()
    model.to(device)
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        for _ in range(max_length):
            # Truncate if too long
            if input_ids.shape[1] > config.max_seq_length - 1:
                input_ids = input_ids[:, -(config.max_seq_length - 1):]
            
            # Get predictions
            logits = model(input_ids)
            next_token_logits = logits[:, -1, :] / temperature
            
            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Stop if EOS token
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated_text

if __name__ == "__main__":
    print("=" * 50)
    print("Tiny Language Model Trainer")
    print("=" * 50)
    
    # Initialize tokenizer
    print("\nInitializing tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load data
    train_loader, val_loader = load_data(tokenizer, config)
    
    # Create model
    print("\nCreating model...")
    model = TinyLanguageModel(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Train model
    train(model, train_loader, val_loader, config, tokenizer)
    
    # Test generation
    print("\n" + "=" * 50)
    print("Testing text generation...")
    print("=" * 50)
    
    prompts = [
        "The history of",
        "In the future,",
        "Once upon a time"
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        generated = generate_text(model, tokenizer, prompt, max_length=50, device=config.device)
        print(f"Generated: {generated}")
        print("-" * 50)