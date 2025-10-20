"""
Flask Web App for Tiny Language Model
Serves your trained model through a simple web interface
"""

from flask import Flask, render_template_string, request, jsonify
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer
import os

app = Flask(__name__)

# Configuration (must match training config)
class Config:
    vocab_size = 50257
    max_seq_length = 128
    embed_dim = 128
    num_layers = 4
    num_heads = 4
    ff_dim = 512
    dropout = 0.1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "tiny_lm_model.pt"

config = Config()

# Model definition (same as training script)
class TinyLanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.position_embedding = nn.Embedding(config.max_seq_length, config.embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        
        self.ln_f = nn.LayerNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
    
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_length = input_ids.shape
        
        positions = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_length)
        
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(positions)
        x = token_embeds + position_embeds
        
        causal_mask = torch.triu(torch.ones(seq_length, seq_length, device=input_ids.device), diagonal=1).bool()
        x = self.transformer(x, mask=causal_mask, src_key_padding_mask=attention_mask)
        
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits

def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.8):
    """Generate text from a prompt"""
    model.eval()
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(config.device)
    
    with torch.no_grad():
        for _ in range(max_length):
            if input_ids.shape[1] > config.max_seq_length - 1:
                input_ids = input_ids[:, -(config.max_seq_length - 1):]
            
            logits = model(input_ids)
            next_token_logits = logits[:, -1, :] / temperature
            
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated_text

# HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Tiny Language Model</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            max-width: 800px;
            width: 100%;
            padding: 40px;
        }
        
        h1 {
            color: #333;
            margin-bottom: 10px;
            font-size: 2em;
        }
        
        .subtitle {
            color: #666;
            margin-bottom: 30px;
            font-size: 0.9em;
        }
        
        .input-group {
            margin-bottom: 20px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            color: #555;
            font-weight: 500;
        }
        
        input[type="text"],
        textarea {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 16px;
            transition: border-color 0.3s;
            font-family: inherit;
        }
        
        input[type="text"]:focus,
        textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .settings {
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .settings > div {
            flex: 1;
        }
        
        input[type="number"],
        input[type="range"] {
            width: 100%;
            padding: 8px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 14px;
        }
        
        .range-container {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .range-value {
            min-width: 40px;
            text-align: center;
            font-weight: 600;
            color: #667eea;
        }
        
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 14px 32px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
            width: 100%;
        }
        
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
        }
        
        button:active {
            transform: translateY(0);
        }
        
        button:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }
        
        .output {
            margin-top: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            display: none;
        }
        
        .output.show {
            display: block;
        }
        
        .output h3 {
            color: #333;
            margin-bottom: 10px;
        }
        
        .output-text {
            color: #555;
            line-height: 1.6;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        
        .loading {
            text-align: center;
            padding: 20px;
            display: none;
        }
        
        .loading.show {
            display: block;
        }
        
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .error {
            background: #fee;
            border-left-color: #f44;
            color: #c00;
        }
        
        .examples {
            margin-top: 20px;
            padding-top: 20px;
            border-top: 1px solid #e0e0e0;
        }
        
        .examples h4 {
            color: #666;
            margin-bottom: 10px;
            font-size: 0.9em;
        }
        
        .example-buttons {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        
        .example-btn {
            background: #f0f0f0;
            color: #555;
            padding: 8px 16px;
            border-radius: 6px;
            font-size: 14px;
            cursor: pointer;
            transition: background 0.2s;
            border: 1px solid #ddd;
        }
        
        .example-btn:hover {
            background: #e0e0e0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Tiny Language Model</h1>
        <p class="subtitle">Your personal AI text generator (~1M parameters)</p>
        
        <div class="input-group">
            <label for="prompt">Enter your prompt:</label>
            <input type="text" id="prompt" placeholder="The future of artificial intelligence is..." />
        </div>
        
        <div class="settings">
            <div>
                <label for="max-length">Max Length:</label>
                <input type="number" id="max-length" value="100" min="10" max="500" />
            </div>
            <div>
                <label for="temperature">Temperature: <span class="range-value" id="temp-value">0.8</span></label>
                <input type="range" id="temperature" min="0.1" max="2.0" step="0.1" value="0.8" 
                       oninput="document.getElementById('temp-value').textContent = this.value" />
            </div>
        </div>
        
        <button id="generate-btn" onclick="generateText()">Generate Text</button>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Generating text...</p>
        </div>
        
        <div class="output" id="output">
            <h3>Generated Text:</h3>
            <div class="output-text" id="output-text"></div>
        </div>
        
        <div class="examples">
            <h4>Try these examples:</h4>
            <div class="example-buttons">
                <span class="example-btn" onclick="setPrompt('The history of')">The history of</span>
                <span class="example-btn" onclick="setPrompt('In the year 2050,')">In the year 2050,</span>
                <span class="example-btn" onclick="setPrompt('Once upon a time')">Once upon a time</span>
                <span class="example-btn" onclick="setPrompt('The secret to happiness is')">The secret to happiness</span>
            </div>
        </div>
    </div>
    
    <script>
        function setPrompt(text) {
            document.getElementById('prompt').value = text;
        }
        
        async function generateText() {
            const prompt = document.getElementById('prompt').value;
            const maxLength = parseInt(document.getElementById('max-length').value);
            const temperature = parseFloat(document.getElementById('temperature').value);
            
            if (!prompt.trim()) {
                alert('Please enter a prompt!');
                return;
            }
            
            const button = document.getElementById('generate-btn');
            const loading = document.getElementById('loading');
            const output = document.getElementById('output');
            const outputText = document.getElementById('output-text');
            
            button.disabled = true;
            loading.classList.add('show');
            output.classList.remove('show');
            
            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        prompt: prompt,
                        max_length: maxLength,
                        temperature: temperature
                    })
                });
                
                const data = await response.json();
                
                if (data.error) {
                    output.classList.add('error');
                    outputText.textContent = 'Error: ' + data.error;
                } else {
                    output.classList.remove('error');
                    outputText.textContent = data.generated_text;
                }
                
                output.classList.add('show');
            } catch (error) {
                output.classList.add('error', 'show');
                outputText.textContent = 'Error: ' + error.message;
            } finally {
                loading.classList.remove('show');
                button.disabled = false;
            }
        }
        
        // Allow Enter key to generate
        document.getElementById('prompt').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                generateText();
            }
        });
    </script>
</body>
</html>
"""

# Load model and tokenizer on startup
print("Loading model and tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

model = TinyLanguageModel(config)

if os.path.exists(config.model_path):
    model.load_state_dict(torch.load(config.model_path, map_location=config.device))
    model.to(config.device)
    model.eval()
    print(f"Model loaded successfully from {config.model_path}")
    print(f"Running on: {config.device}")
else:
    print(f"ERROR: Model file '{config.model_path}' not found!")
    print("Please train the model first using the training script.")

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        prompt = data.get('prompt', '')
        max_length = data.get('max_length', 100)
        temperature = data.get('temperature', 0.8)
        
        if not prompt:
            return jsonify({'error': 'Prompt is required'}), 400
        
        # Generate text
        generated = generate_text(model, tokenizer, prompt, max_length, temperature)
        
        return jsonify({
            'generated_text': generated,
            'prompt': prompt
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*50)
    print("Starting Tiny Language Model Web Server")
    print("="*50)
    print("\nOpen your browser and go to: http://localhost:5000")
    print("Press Ctrl+C to stop the server\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)