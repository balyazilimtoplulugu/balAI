"""
BAL AI - Bornova Anadolu Lisesi Yapay Zeka Modeli
Flask web uygulaması - Dual Model Support
COMPLETE VERSION - Ready to deploy!
"""

from flask import Flask, render_template_string, request, jsonify, send_from_directory
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer
import os
import gc
import logging
from datetime import datetime
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static', static_url_path='/static')

# Yapılandırma
class Config:
    vocab_size = 50257
    max_seq_length = 128
    embed_dim = 128
    num_layers = 4
    num_heads = 4
    ff_dim = 512
    dropout = 0.1
    device = "cpu"  # Force CPU for Render
    
    # Model paths
    models = {
        "wikitext2": {
            "path": "tiny_lm_model.pt",
            "name": "WikiText-2 (İlk Model)",
            "description": "5MB veri ile eğitildi",
            "params": "13M parametre"
        },
        "wikitext103": {
            "path": "tiny_lm_model_wikitext103_best.pt",
            "name": "WikiText-103 (Gelişmiş Model)",
            "description": "500MB veri ile eğitildi",
            "params": "13M parametre"
        }
    }
    default_model = "wikitext103"

config = Config()

# Model tanımı
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

# Global model cache
loaded_models = {}
model_lock = threading.Lock()

def load_model(model_key):
    """Load a model into memory"""
    with model_lock:
        if model_key in loaded_models:
            return loaded_models[model_key]
        
        model_path = config.models[model_key]["path"]
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return None
        
        try:
            model = TinyLanguageModel(config)
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.to('cpu')
            model.eval()
            loaded_models[model_key] = model
            return model
        except Exception as e:
            logger.error(f"Error loading model {model_key}: {str(e)}")
            return None

def generate_text(model, tokenizer, prompt, max_length=50, temperature=0.8):
    """Metinden devam et"""
    model.eval()
    
    # Use a shorter sequence to save memory
    max_length = min(max_length, 30)
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(config.device)
    
    with torch.no_grad():
        for i in range(max_length):
            if i % 10 == 0:
                gc.collect()
            
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
    
    gc.collect()
    return generated_text

# HTML şablonu
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="tr">
<head>
    <title>BAL AI - Bornova Anadolu Lisesi Yapay Zeka</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    
    <!-- SEO Meta Tags -->
    <meta name="description" content="Bornova Anadolu Lisesi öğrencileri tarafından eğitilmiş yapay zeka dil modelleri.">
    <meta name="keywords" content="Bornova Anadolu Lisesi, BAL, yapay zeka, AI, dil modeli, öğrenci projesi, makine öğrenmesi">
    <meta name="author" content="Bornova Anadolu Lisesi Yazılım Topluluğu">
    
    <!-- Open Graph / Facebook -->
    <meta property="og:type" content="website">
    <meta property="og:url" content="https://balai-5tqa.onrender.com">
    <meta property="og:title" content="BAL AI - Bornova Anadolu Lisesi Yapay Zeka">
    <meta property="og:description" content="Bornova Anadolu Lisesi öğrencileri tarafından eğitilmiş yapay zeka dil modelleri.">
    <meta property="og:image" content="https://balai-5tqa.onrender.com/logo.png">

    <!-- Twitter -->
    <meta name="twitter:card" content="summary_large_image">
    <meta name="twitter:title" content="BAL AI - Bornova Anadolu Lisesi Yapay Zeka">
    <meta name="twitter:description" content="Bornova Anadolu Lisesi öğrencileri tarafından eğitilmiş yapay zeka dil modelleri.">
    <meta name="twitter:image" content="https://balai-5tqa.onrender.com/logo.png">
    
    <!-- Favicon -->
    <link rel="shortcut icon" href="/favicon.ico">
    <link rel="icon" href="/favicon.ico" type="image/x-icon">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
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
            max-width: 900px;
            width: 100%;
            padding: 40px;
            animation: fadeIn 0.5s ease-in;
        }
        
        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .header {
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 30px;
            border-bottom: 3px solid #fee2e2;
        }
        
        .logo-section {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .logo-icon {
            font-size: 3em;
            background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        h1 {
            color: #991b1b;
            font-size: 2.5em;
            font-weight: 700;
            letter-spacing: -1px;
        }
        
        .subtitle {
            color: #666;
            margin-top: 10px;
            font-size: 1.1em;
        }
        
        .model-selector {
            background: #fef2f2;
            border: 2px solid #fecaca;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 30px;
        }
        
        .model-selector h3 {
            color: #991b1b;
            font-size: 1.1em;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .model-options {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }
        
        .model-option {
            background: white;
            border: 2px solid #fecaca;
            border-radius: 10px;
            padding: 15px;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .model-option:hover {
            border-color: #dc2626;
            box-shadow: 0 4px 12px rgba(220, 38, 38, 0.2);
        }
        
        .model-option.active {
            border-color: #dc2626;
            background: linear-gradient(to bottom right, #fef2f2, #fff);
            box-shadow: 0 4px 15px rgba(220, 38, 38, 0.3);
        }
        
        .model-option input[type="radio"] {
            margin-right: 10px;
        }
        
        .model-name {
            font-weight: 700;
            color: #991b1b;
            font-size: 1.05em;
            margin-bottom: 5px;
        }
        
        .model-desc {
            font-size: 0.85em;
            color: #666;
        }
        
        .badge {
            display: inline-block;
            background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
            color: white;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.75em;
            font-weight: 600;
            margin-top: 5px;
        }
        
        .input-section {
            margin-bottom: 30px;
        }
        
        label {
            display: block;
            margin-bottom: 10px;
            color: #991b1b;
            font-weight: 600;
            font-size: 1.05em;
        }
        
        label i {
            margin-right: 8px;
        }
        
        input[type="text"] {
            width: 100%;
            padding: 15px;
            border: 2px solid #fecaca;
            border-radius: 12px;
            font-size: 16px;
            transition: all 0.3s;
            font-family: inherit;
            background: #fef2f2;
        }
        
        input[type="text"]:focus {
            outline: none;
            border-color: #dc2626;
            background: white;
            box-shadow: 0 0 0 4px rgba(220, 38, 38, 0.1);
        }
        
        .settings {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 25px;
            padding: 20px;
            background: #fef2f2;
            border-radius: 12px;
            border: 2px solid #fecaca;
        }
        
        .setting-group {
            display: flex;
            flex-direction: column;
        }
        
        .setting-label {
            color: #991b1b;
            font-weight: 600;
            margin-bottom: 8px;
            font-size: 0.95em;
        }
        
        input[type="number"] {
            padding: 10px;
            border: 2px solid #fecaca;
            border-radius: 8px;
            font-size: 15px;
            background: white;
        }
        
        input[type="range"] {
            width: 100%;
            height: 8px;
            border-radius: 5px;
            background: #fecaca;
            outline: none;
            -webkit-appearance: none;
        }
        
        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
            cursor: pointer;
            box-shadow: 0 2px 8px rgba(220, 38, 38, 0.4);
        }
        
        input[type="range"]::-moz-range-thumb {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
            cursor: pointer;
            box-shadow: 0 2px 8px rgba(220, 38, 38, 0.4);
            border: none;
        }
        
        .range-value {
            display: inline-block;
            min-width: 45px;
            text-align: center;
            font-weight: 700;
            color: #dc2626;
            background: white;
            padding: 4px 10px;
            border-radius: 6px;
            margin-left: 10px;
        }
        
        .generate-btn {
            width: 100%;
            background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
            color: white;
            border: none;
            padding: 18px 32px;
            border-radius: 12px;
            font-size: 18px;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.3s;
            box-shadow: 0 6px 20px rgba(220, 38, 38, 0.4);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
        }
        
        .generate-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(220, 38, 38, 0.5);
        }
        
        .generate-btn:active {
            transform: translateY(0);
        }
        
        .generate-btn:disabled {
            background: #d1d5db;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
        
        .loading {
            text-align: center;
            padding: 30px;
            display: none;
        }
        
        .loading.show {
            display: block;
        }
        
        .spinner {
            border: 4px solid #fecaca;
            border-top: 4px solid #dc2626;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .loading-text {
            color: #991b1b;
            font-weight: 600;
            font-size: 1.1em;
        }
        
        .output {
            margin-top: 30px;
            padding: 25px;
            background: linear-gradient(to bottom right, #fef2f2, #fff);
            border-radius: 12px;
            border-left: 5px solid #dc2626;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            display: none;
            animation: slideIn 0.4s ease-out;
        }
        
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateX(-20px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        
        .output.show {
            display: block;
        }
        
        .output h3 {
            color: #991b1b;
            margin-bottom: 15px;
            font-size: 1.3em;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .output-text {
            color: #374151;
            line-height: 1.8;
            white-space: pre-wrap;
            word-wrap: break-word;
            font-size: 1.05em;
        }
        
        .error {
            background: #fee2e2;
            border-left-color: #dc2626;
        }
        
        .error .output-text {
            color: #991b1b;
            font-weight: 600;
        }
        
        .examples {
            margin-top: 30px;
            padding-top: 25px;
            border-top: 2px solid #fecaca;
        }
        
        .examples h4 {
            color: #991b1b;
            margin-bottom: 15px;
            font-size: 1.1em;
            font-weight: 600;
        }
        
        .example-buttons {
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
        }
        
        .example-btn {
            background: #fef2f2;
            color: #991b1b;
            padding: 10px 18px;
            border-radius: 8px;
            font-size: 14px;
            cursor: pointer;
            transition: all 0.2s;
            border: 2px solid #fecaca;
            font-weight: 500;
        }
        
        .example-btn:hover {
            background: #fee2e2;
            border-color: #dc2626;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(220, 38, 38, 0.2);
        }
        
        .footer {
            text-align: center;
            margin-top: 40px;
            padding-top: 25px;
            border-top: 2px solid #fecaca;
            color: #991b1b;
            font-size: 0.95em;
        }
        
        .footer i {
            color: #dc2626;
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 25px;
            }
            
            h1 {
                font-size: 2em;
            }
            
            .settings, .model-options {
                grid-template-columns: 1fr;
            }
            
            .example-buttons {
                flex-direction: column;
            }
            
            .example-btn {
                width: 100%;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="logo-section">
                <i class="fas fa-brain logo-icon"></i>
                <h1>BAL AI</h1>
            </div>
            <p class="subtitle">Bornova Anadolu Lisesi Yapay Zeka Modeli</p>
        </div>

        <div class="model-selector">
            <h3>
                <i class="fas fa-robot"></i>
                Model Seçimi:
            </h3>
            <div class="model-options">
                <label class="model-option" for="model-wikitext2">
                    <input type="radio" name="model" id="model-wikitext2" value="wikitext2">
                    <div>
                        <div class="model-name">WikiText-2</div>
                        <div class="model-desc">İlk deneysel model</div>
                        <span class="badge">5MB veri</span>
                    </div>
                </label>
                <label class="model-option active" for="model-wikitext103">
                    <input type="radio" name="model" id="model-wikitext103" value="wikitext103" checked>
                    <div>
                        <div class="model-name">WikiText-103</div>
                        <div class="model-desc">Gelişmiş model</div>
                        <span class="badge">500MB veri</span>
                    </div>
                </label>
            </div>
        </div>

        <div class="input-section">
            <label for="prompt">
                <i class="fas fa-pencil-alt"></i>
                Başlangıç Metni Girin:
            </label>
            <input 
                type="text" 
                id="prompt" 
                placeholder="Örn: The history of..." 
                autocomplete="off"
            />
        </div>

        <div class="settings">
            <div class="setting-group">
                <div class="setting-label">
                    <i class="fas fa-text-width"></i>
                    Maksimum Uzunluk:
                </div>
                <input type="number" id="max-length" value="20" min="5" max="50" />
            </div>
            <div class="setting-group">
                <div class="setting-label">
                    <i class="fas fa-temperature-high"></i>
                    Yaratıcılık Seviyesi:
                    <span class="range-value" id="temp-value">0.8</span>
                </div>
                <input 
                    type="range" 
                    id="temperature" 
                    min="0.1" 
                    max="2.0" 
                    step="0.1" 
                    value="0.8" 
                    oninput="document.getElementById('temp-value').textContent = this.value"
                />
            </div>
        </div>

        <button class="generate-btn" onclick="generateText()">
            <i class="fas fa-magic"></i>
            Metin Üret
        </button>

        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p class="loading-text">Metin üretiliyor...</p>
        </div>

        <div class="output" id="output">
            <h3>
                <i class="fas fa-file-alt"></i>
                Üretilen Metin:
            </h3>
            <div class="output-text" id="output-text"></div>
        </div>

        <div class="examples">
            <h4>
                <i class="fas fa-lightbulb"></i>
                Örnek Başlangıçlar:
            </h4>
            <div class="example-buttons">
                <span class="example-btn" onclick="setPrompt('The history of')">
                    <i class="fas fa-book"></i> The history of
                </span>
                <span class="example-btn" onclick="setPrompt('In the United States,')">
                    <i class="fas fa-flag-usa"></i> In the United States,
                </span>
                <span class="example-btn" onclick="setPrompt('Once upon a time')">
                    <i class="fas fa-scroll"></i> Once upon a time
                </span>
                <span class="example-btn" onclick="setPrompt('The first season of')">
                    <i class="fas fa-tv"></i> The first season of
                </span>
                <span class="example-btn" onclick="setPrompt('In the future,')">
                    <i class="fas fa-rocket"></i> In the future,
                </span>
            </div>
        </div>

        <div class="footer">
            <p>
                <i class="fas fa-heart"></i>
                Bornova Anadolu Lisesi Yazılım Topluluğu tarafından geliştirildi
            </p>
            <p style="margin-top: 8px; font-size: 0.85em; color: #666;">
                13 Milyon parametre • WikiText veri setleri ile eğitildi
            </p>
        </div>
    </div>

    <script>
        // Model selection handling
        document.querySelectorAll('.model-option').forEach(option => {
            option.addEventListener('click', function() {
                document.querySelectorAll('.model-option').forEach(o => o.classList.remove('active'));
                this.classList.add('active');
                this.querySelector('input[type="radio"]').checked = true;
            });
        });

        function setPrompt(text) {
            document.getElementById('prompt').value = text;
            document.getElementById('prompt').focus();
        }

        async function generateText() {
            const prompt = document.getElementById('prompt').value;
            const maxLength = parseInt(document.getElementById('max-length').value);
            const temperature = parseFloat(document.getElementById('temperature').value);
            const selectedModel = document.querySelector('input[name="model"]:checked').value;

            if (!prompt.trim()) {
                alert('Lütfen bir başlangıç metni girin!');
                return;
            }

            const button = document.querySelector('.generate-btn');
            const loading = document.getElementById('loading');
            const output = document.getElementById('output');
            const outputText = document.getElementById('output-text');

            button.disabled = true;
            button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Üretiliyor...';
            loading.classList.add('show');
            output.classList.remove('show');

            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 60000);

            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        prompt: prompt,
                        max_length: maxLength,
                        temperature: temperature,
                        model: selectedModel
                    }),
                    signal: controller.signal
                });

                clearTimeout(timeoutId);

                if (!response.ok) {
                    const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
                    throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
                }

                const data = await response.json();

                if (data.error) {
                    output.classList.add('error');
                    outputText.textContent = '❌ Hata: ' + data.error;
                } else {
                    output.classList.remove('error');
                    outputText.textContent = data.generated_text;
                }

                output.classList.add('show');
            } catch (error) {
                output.classList.add('error', 'show');
                if (error.name === 'AbortError') {
                    outputText.textContent = '❌ İstek zaman aşımına uğradı. Lütfen daha sonra tekrar deneyin.';
                } else {
                    outputText.textContent = '❌ Hata: ' + error.message;
                }
                console.error('Error:', error);
            } finally {
                clearTimeout(timeoutId);
                loading.classList.remove('show');
                button.disabled = false;
                button.innerHTML = '<i class="fas fa-magic"></i> Metin Üret';
            }
        }

        // Enter tuşu ile üret
        document.getElementById('prompt').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                generateText();
            }
        });
    </script>
</body>
</html>
"""

# Initialize tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Preload default model on startup
default_model = load_model(config.default_model)

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/favicon.ico')
def favicon():
    return send_from_directory('static', 'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/logo.png')
def logo():
    return send_from_directory('static', 'logo.png', mimetype='image/png')

@app.route('/generate', methods=['POST'])
def generate():
    start_time = datetime.now()
    
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'JSON verisi alınamadı'}), 400
            
        prompt = data.get('prompt', '')
        max_length = min(data.get('max_length', 20), 30)
        temperature = data.get('temperature', 0.8)
        model_key = data.get('model', config.default_model)
        
        if not prompt:
            return jsonify({'error': 'Prompt gerekli'}), 400
        
        # Validate model key
        if model_key not in config.models:
            return jsonify({'error': f'Geçersiz model: {model_key}'}), 400
        
        # Load model if not already loaded
        model = load_model(model_key)
        if model is None:
            return jsonify({'error': f'Model yüklenemedi: {config.models[model_key]["name"]}'}), 500
        
        
        # Generate text with timeout protection
        result = {'text': None, 'error': None}
        
        def generate_with_timeout():
            try:
                result['text'] = generate_text(model, tokenizer, prompt, max_length, temperature)
            except Exception as e:
                result['error'] = str(e)
        
        thread = threading.Thread(target=generate_with_timeout)
        thread.daemon = True
        thread.start()
        thread.join(timeout=25)
        
        if thread.is_alive():
            return jsonify({'error': 'Metin üretimi çok uzun sürdü. Daha kısa bir uzunluk deneyin.'}), 408
        
        if result['error']:
            return jsonify({'error': result['error']}), 500
        
        if result['text'] is None:
            return jsonify({'error': 'Metin üretilemedi'}), 500
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        return jsonify({
            'generated_text': result['text'],
            'prompt': prompt,
            'model': model_key
        })
    
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    try:
        models_status = {}
        for key, model_info in config.models.items():
            models_status[key] = {
                'name': model_info['name'],
                'available': os.path.exists(model_info['path']),
                'loaded': key in loaded_models
            }
        
        return jsonify({
            'status': 'ok',
            'models': models_status,
            'device': config.device
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)