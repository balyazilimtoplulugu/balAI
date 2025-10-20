"""
Tiny Language Model Trainer (~13M parameters)
This script trains a small transformer model from scratch on the WikiText-103 dataset.
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
    num_epochs = 5  # Increased for larger dataset
    save_every = 1000
    eval_every = 500
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Paths
    model_save_path = "tiny_lm_model_wikitext103.pt"

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
    """Load WikiText-103 dataset (much larger - 500MB vs 5MB)"""
    print("=" * 60)
    print("DOWNLOADING WIKITEXT-103 DATASET")
    print("=" * 60)
    print("This is 100x larger than WikiText-2 (~500MB)")
    print("Download may take a few minutes...")
    print()
    
    dataset = load_dataset("wikitext", "wikitext-103-v1")
    
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"Training examples: {len(dataset['train']['text']):,}")
    print(f"Validation examples: {len(dataset['validation']['text']):,}")
    print()
    
    print("Preparing training data...")
    train_dataset = TextDataset(dataset["train"]["text"], tokenizer, config.max_seq_length)
    
    print("Preparing validation data...")
    val_dataset = TextDataset(dataset["validation"]["text"], tokenizer, config.max_seq_length)
    
    print(f"\nProcessed training sequences: {len(train_dataset):,}")
    print(f"Processed validation sequences: {len(val_dataset):,}")
    print()
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=2)
    
    return train_loader, val_loader

def calculate_perplexity(model, dataloader, device, max_batches=100):
    """Calculate perplexity on a dataset (limited batches for speed)"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
                
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
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=len(train_loader) * config.num_epochs
    )
    
    model.to(config.device)
    global_step = 0
    best_perplexity = float('inf')
    
    print("\n" + "=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Device: {config.device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Training batches per epoch: {len(train_loader):,}")
    print(f"Estimated time per epoch: ~2-3 hours (on GPU)")
    print("=" * 60)
    print()
    
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
            scheduler.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Evaluate periodically
            if global_step % config.eval_every == 0:
                perplexity = calculate_perplexity(model, val_loader, config.device)
                print(f"\n[Step {global_step}] Validation Perplexity: {perplexity:.2f}")
                
                # Save best model
                if perplexity < best_perplexity:
                    best_perplexity = perplexity
                    torch.save(model.state_dict(), config.model_save_path.replace('.pt', '_best.pt'))
                    print(f"✓ New best model saved! (Perplexity: {perplexity:.2f})")
                
                model.train()
            
            # Save checkpoint
            if global_step % config.save_every == 0:
                checkpoint_path = config.model_save_path.replace('.pt', f'_step{global_step}.pt')
                torch.save(model.state_dict(), checkpoint_path)
                print(f"\n✓ Checkpoint saved: {checkpoint_path}")
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"\nEpoch {epoch+1} Complete - Average Loss: {avg_loss:.4f}")
        
        # Full validation at end of epoch
        val_perplexity = calculate_perplexity(model, val_loader, config.device, max_batches=200)
        print(f"End of Epoch Validation Perplexity: {val_perplexity:.2f}\n")
    
    # Final save
    torch.save(model.state_dict(), config.model_save_path)
    print(f"\n{'=' * 60}")
    print("TRAINING COMPLETE!")
    print(f"{'=' * 60}")
    print(f"Final model saved to: {config.model_save_path}")
    print(f"Best model saved to: {config.model_save_path.replace('.pt', '_best.pt')}")
    print(f"Best validation perplexity: {best_perplexity:.2f}")
    print()

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
    print("\n" + "=" * 60)
    print("TINY LANGUAGE MODEL TRAINER - WIKITEXT-103")
    print("=" * 60)
    print("Training a 13M parameter model on 500MB of Wikipedia text")
    print("Expected improvement over WikiText-2: SIGNIFICANT!")
    print("=" * 60)
    print()
    
    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load data
    train_loader, val_loader = load_data(tokenizer, config)
    
    # Create model
    print("Creating model...")
    model = TinyLanguageModel(config)
    print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters\n")
    
    # Train model
    train(model, train_loader, val_loader, config, tokenizer)
    
    # Test generation
    print("=" * 60)
    print("TESTING TEXT GENERATION")
    print("=" * 60)
    print()
    
    prompts = [
        "The history of the United States",
        "In the future, artificial intelligence",
        "Once upon a time in a distant land"
    ]
    
    for prompt in prompts:
        print(f"Prompt: {prompt}")
        generated = generate_text(model, tokenizer, prompt, max_length=80, device=config.device)
        print(f"Generated: {generated}")
        print("-" * 60)
        print()