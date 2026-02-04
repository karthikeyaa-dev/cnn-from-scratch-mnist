from flask import Flask, request, jsonify, render_template, send_file
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import io
import base64
import json
import os
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any
import warnings
import uuid
import time
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ============================================================================
# MODEL DEFINITION (MUST MATCH TRAINING)
# ============================================================================

class MNISTCNN(nn.Module):
    """CNN model for MNIST classification."""
    
    def __init__(self, dropout_rate: float = 0.3):
        super(MNISTCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate/2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate)
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate/2),
            nn.Linear(128, 10)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc_layers(x)
        return x

# ============================================================================
# MODEL LOADING (FIXED FOR PYTORCH 2.6)
# ============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = MNISTCNN(dropout_rate=0.3).to(device)

# Try to load the best model with safe loading
model_paths = ['best_mnist_cnn_model.pth', 'mnist_cnn_model.pth', 'mnist_cnn_best.pth']
loaded = False

for model_path in model_paths:
    if os.path.exists(model_path):
        try:
            # For PyTorch 2.6, we need to handle weights_only parameter
            import torch.serialization
            # Add safe globals for numpy
            torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
            
            # Try with weights_only=False first (for backward compatibility)
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            
            # Load model state dict
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # Try direct loading
                model.load_state_dict(checkpoint)
            
            print(f"✓ Model loaded from {model_path}")
            loaded = True
            
            # Print model info if available
            if 'test_metrics' in checkpoint:
                test_acc = checkpoint['test_metrics'].get('accuracy', 'N/A')
                print(f"✓ Model accuracy: {test_acc:.4f}" if isinstance(test_acc, (int, float)) else f"✓ Model accuracy: {test_acc}")
            break
            
        except Exception as e:
            print(f"✗ Error loading {model_path}: {e}")
            # Try alternative loading method
            try:
                checkpoint = torch.load(model_path, map_location=device)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                print(f"✓ Model loaded (alternative method) from {model_path}")
                loaded = True
                break
            except Exception as e2:
                print(f"✗ Alternative loading also failed: {e2}")
                continue

if not loaded:
    print("\n⚠  No pre-trained model found. Creating new model...")
    print("⚠  Train a model first or place a model file in the directory.")
    print("⚠  Expected model files: best_mnist_cnn_model.pth or mnist_cnn_model.pth")

model.eval()

# Create directories for debug images
os.makedirs('static/debug', exist_ok=True)
os.makedirs('static/preview', exist_ok=True)

# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

def save_debug_image(img_array: np.ndarray, stage: str, request_id: str):
    """Save debug images for troubleshooting."""
    if not app.config.get('DEBUG', False):
        return None
    
    debug_path = f'static/debug/{request_id}_{stage}.png'
    plt.figure(figsize=(3, 3))
    plt.imshow(img_array, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(debug_path, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close()
    return debug_path

def preprocess_image_basic(image_bytes: bytes, request_id: str = None) -> torch.Tensor:
    """
    Basic preprocessing - just resize and normalize.
    """
    try:
        # Open and convert to grayscale
        img = Image.open(io.BytesIO(image_bytes)).convert('L')
        
        # Resize to 28x28
        img = img.resize((28, 28), Image.Resampling.BILINEAR)
        img_array = np.array(img, dtype=np.float32)
        
        if request_id:
            save_debug_image(img_array, '00_original', request_id)
        
        # Auto-invert: MNIST has white digits on black background
        if np.mean(img_array) > 127:
            img_array = 255 - img_array
            if request_id:
                save_debug_image(img_array, '01_inverted', request_id)
        
        # Normalize to [0, 1]
        img_array = img_array / 255.0
        
        if request_id:
            save_debug_image(img_array, '02_normalized', request_id)
        
        # Add channel dimension
        img_array = img_array.reshape(1, 28, 28)
        
        return torch.from_numpy(img_array)
    
    except Exception as e:
        raise ValueError(f"Basic preprocessing failed: {str(e)}")

def preprocess_image_enhanced(image_bytes: bytes, request_id: str = None) -> torch.Tensor:
    """
    Enhanced preprocessing for real-world handwritten digits.
    """
    try:
        # Open image
        img = Image.open(io.BytesIO(image_bytes)).convert('L')
        
        # Convert to numpy
        img_array = np.array(img, dtype=np.float32)
        
        if request_id:
            save_debug_image(img_array, '00_original', request_id)
        
        # Step 1: Determine background and invert if needed
        edge_pixels = np.concatenate([
            img_array[0, :],  # top
            img_array[-1, :], # bottom
            img_array[:, 0],  # left
            img_array[:, -1]  # right
        ])
        bg_mean = np.mean(edge_pixels)
        
        # Invert if background is dark (digit should be white)
        if bg_mean < 128:
            img_array = 255 - img_array
            if request_id:
                save_debug_image(img_array, '01_inverted', request_id)
        
        # Step 2: Simple thresholding
        img_array = np.where(img_array > 50, 255, 0).astype(np.float32)
        
        if request_id:
            save_debug_image(img_array, '02_thresholded', request_id)
        
        # Step 3: Find bounding box of the digit
        rows = np.any(img_array > 10, axis=1)
        cols = np.any(img_array > 10, axis=0)
        
        if np.any(rows) and np.any(cols):
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            
            # Add padding
            h, w = y_max - y_min, x_max - x_min
            pad = max(5, int(0.15 * max(h, w)))
            
            y_min = max(0, y_min - pad)
            y_max = min(img_array.shape[0] - 1, y_max + pad)
            x_min = max(0, x_min - pad)
            x_max = min(img_array.shape[1] - 1, x_max + pad)
            
            # Crop the digit
            digit = img_array[y_min:y_max+1, x_min:x_max+1]
            
            if request_id:
                save_debug_image(digit, '03_cropped', request_id)
            
            # Step 4: Resize while preserving aspect ratio
            h, w = digit.shape
            
            # Calculate scale to fit in 20x20
            scale = 20.0 / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            
            # Ensure minimum size
            new_h = max(new_h, 4)
            new_w = max(new_w, 4)
            
            # Resize using PIL
            digit_pil = Image.fromarray(digit.astype(np.uint8))
            digit_pil = digit_pil.resize((new_w, new_h), Image.Resampling.BILINEAR)
            digit = np.array(digit_pil).astype(np.float32)
            
            if request_id:
                save_debug_image(digit, '04_resized', request_id)
            
            # Step 5: Center in 28x28 canvas
            canvas = np.zeros((28, 28), dtype=np.float32)
            y_offset = (28 - new_h) // 2
            x_offset = (28 - new_w) // 2
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = digit
            
            if request_id:
                save_debug_image(canvas, '05_padded', request_id)
            
            # Step 6: Center of mass adjustment
            try:
                cy, cx = ndimage.center_of_mass(canvas)
                shift_y = int(round(14 - cy))
                shift_x = int(round(14 - cx))
                
                if abs(shift_y) > 0 or abs(shift_x) > 0:
                    canvas = ndimage.shift(canvas, shift=(shift_y, shift_x), 
                                          mode='constant', cval=0)
                
                if request_id:
                    save_debug_image(canvas, '06_centered', request_id)
            except:
                pass  # Skip centering if it fails
            
            img_array = canvas
        
        # Step 7: Normalize to [0, 1]
        img_array = img_array / 255.0
        
        if request_id:
            save_debug_image(img_array, '07_final', request_id)
        
        # Add channel dimension
        img_array = img_array.reshape(1, 28, 28)
        
        return torch.from_numpy(img_array)
    
    except Exception as e:
        print(f"Enhanced preprocessing failed: {e}")
        # Fall back to basic preprocessing
        return preprocess_image_basic(image_bytes, request_id)

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def predict_digit(image_bytes: bytes, preprocessing_mode: str = "enhanced", 
                  request_id: str = None) -> Dict[str, Any]:
    """
    Predict digit from image bytes.
    """
    start_time = time.time()
    
    try:
        # Choose preprocessing method
        if preprocessing_mode == "basic":
            img_tensor = preprocess_image_basic(image_bytes, request_id)
        else:  # enhanced (default)
            img_tensor = preprocess_image_enhanced(image_bytes, request_id)
        
        # Add batch dimension and move to device
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            logits = model(img_tensor)
            probabilities = F.softmax(logits, dim=1)[0]
            
            # Get top predictions
            top_probs, top_indices = torch.topk(probabilities, 3)
            
            prediction = int(top_indices[0].item())
            confidence = float(top_probs[0].item())
            
            # Prepare top predictions
            top_predictions = []
            for prob, idx in zip(top_probs, top_indices):
                top_predictions.append({
                    "digit": int(idx.item()),
                    "confidence": float(prob.item()),
                    "probability": float(prob.item())
                })
            
            # All probabilities
            all_probs = probabilities.cpu().numpy().tolist()
            
            # Calculate prediction time
            inference_time = (time.time() - start_time) * 1000  # in milliseconds
        
        # Create result dictionary
        result = {
            "success": True,
            "prediction": prediction,
            "confidence": confidence,
            "top_predictions": top_predictions,
            "all_probabilities": all_probs,
            "inference_time_ms": round(inference_time, 2),
            "preprocessing_mode": preprocessing_mode,
            "model_confidence": "high" if confidence > 0.9 else "medium" if confidence > 0.7 else "low"
        }
        
        # Add debug info if debug mode is enabled
        if app.config.get('DEBUG', False) and request_id:
            result["debug_info"] = {
                "request_id": request_id,
                "debug_images_available": True
            }
        
        return result
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "prediction": None,
            "confidence": 0.0,
            "inference_time_ms": round((time.time() - start_time) * 1000, 2)
        }

# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route('/')
def home():
    """Home page with upload form."""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>MNIST Digit Recognition API</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            
            body {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 20px;
            }
            
            .container {
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                padding: 40px;
                max-width: 800px;
                width: 100%;
            }
            
            .header {
                text-align: center;
                margin-bottom: 30px;
            }
            
            h1 {
                color: #333;
                font-size: 2.5em;
                margin-bottom: 10px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            
            .subtitle {
                color: #666;
                font-size: 1.1em;
                margin-bottom: 20px;
            }
            
            .upload-area {
                border: 3px dashed #667eea;
                border-radius: 15px;
                padding: 40px;
                text-align: center;
                margin-bottom: 30px;
                cursor: pointer;
                transition: all 0.3s ease;
            }
            
            .upload-area:hover {
                border-color: #764ba2;
                background: #f9f9ff;
            }
            
            .upload-icon {
                font-size: 60px;
                color: #667eea;
                margin-bottom: 20px;
            }
            
            .upload-text {
                font-size: 1.2em;
                color: #555;
                margin-bottom: 10px;
            }
            
            .upload-hint {
                color: #888;
                font-size: 0.9em;
            }
            
            .file-input {
                display: none;
            }
            
            .options {
                margin-bottom: 30px;
            }
            
            .option-group {
                margin-bottom: 15px;
            }
            
            label {
                display: block;
                margin-bottom: 8px;
                color: #555;
                font-weight: 500;
            }
            
            select {
                width: 100%;
                padding: 12px;
                border: 2px solid #ddd;
                border-radius: 8px;
                font-size: 1em;
                transition: border-color 0.3s ease;
            }
            
            select:focus {
                border-color: #667eea;
                outline: none;
            }
            
            .buttons {
                display: flex;
                gap: 15px;
                margin-bottom: 30px;
            }
            
            button {
                flex: 1;
                padding: 15px;
                border: none;
                border-radius: 10px;
                font-size: 1.1em;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
            }
            
            #predict-btn {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            
            #predict-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
            }
            
            #clear-btn {
                background: #f1f1f1;
                color: #555;
            }
            
            #clear-btn:hover {
                background: #e1e1e1;
            }
            
            #result {
                display: none;
                background: #f9f9ff;
                border-radius: 15px;
                padding: 30px;
                margin-top: 30px;
            }
            
            .result-header {
                font-size: 1.5em;
                color: #333;
                margin-bottom: 20px;
                text-align: center;
            }
            
            .prediction-card {
                background: white;
                border-radius: 12px;
                padding: 25px;
                margin-bottom: 20px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.05);
                text-align: center;
            }
            
            .prediction-digit {
                font-size: 5em;
                font-weight: bold;
                color: #667eea;
                margin: 20px 0;
            }
            
            .confidence-bar {
                height: 10px;
                background: #eee;
                border-radius: 5px;
                margin: 20px 0;
                overflow: hidden;
            }
            
            .confidence-fill {
                height: 100%;
                background: linear-gradient(90deg, #4CAF50, #8BC34A);
                border-radius: 5px;
                transition: width 1s ease;
            }
            
            .confidence-text {
                font-size: 1.2em;
                color: #555;
                margin-top: 10px;
            }
            
            .top-predictions {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 15px;
                margin-top: 20px;
            }
            
            .top-prediction {
                background: white;
                border-radius: 10px;
                padding: 15px;
                text-align: center;
                box-shadow: 0 3px 10px rgba(0,0,0,0.05);
            }
            
            .top-digit {
                font-size: 2em;
                font-weight: bold;
                color: #764ba2;
                margin-bottom: 5px;
            }
            
            .top-confidence {
                color: #666;
                font-size: 0.9em;
            }
            
            .error {
                background: #ffebee;
                color: #c62828;
                padding: 15px;
                border-radius: 10px;
                margin-top: 20px;
                text-align: center;
            }
            
            .loading {
                text-align: center;
                padding: 30px;
                color: #667eea;
                font-size: 1.2em;
            }
            
            .loading::after {
                content: '';
                animation: dots 1.5s infinite;
            }
            
            @keyframes dots {
                0%, 20% { content: '.'; }
                40% { content: '..'; }
                60%, 100% { content: '...'; }
            }
            
            @media (max-width: 600px) {
                .container {
                    padding: 20px;
                }
                
                h1 {
                    font-size: 2em;
                }
                
                .buttons {
                    flex-direction: column;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>✍️ MNIST Digit Recognition</h1>
                <div class="subtitle">Upload an image of a handwritten digit (0-9) for classification</div>
            </div>
            
            <div class="upload-area" id="uploadArea">
                <div class="upload-icon">📁</div>
                <div class="upload-text">Click to upload or drag & drop</div>
                <div class="upload-hint">Supported: JPG, PNG, JPEG (Max 5MB)</div>
                <input type="file" id="fileInput" class="file-input" accept="image/*">
            </div>
            
            <div class="options">
                <div class="option-group">
                    <label for="preprocessing">Preprocessing Mode:</label>
                    <select id="preprocessing">
                        <option value="enhanced">Enhanced (Recommended)</option>
                        <option value="basic">Basic</option>
                    </select>
                </div>
            </div>
            
            <div class="buttons">
                <button id="predict-btn">🔍 Predict Digit</button>
                <button id="clear-btn">🗑️ Clear</button>
            </div>
            
            <div id="result">
                <!-- Results will be inserted here -->
            </div>
        </div>
        
        <script>
            const uploadArea = document.getElementById('uploadArea');
            const fileInput = document.getElementById('fileInput');
            const preprocessingSelect = document.getElementById('preprocessing');
            const resultDiv = document.getElementById('result');
            
            let currentFile = null;
            
            // File upload handling
            uploadArea.addEventListener('click', () => fileInput.click());
            
            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.style.borderColor = '#764ba2';
                uploadArea.style.background = '#f0f0ff';
            });
            
            uploadArea.addEventListener('dragleave', () => {
                uploadArea.style.borderColor = '#667eea';
                uploadArea.style.background = '';
            });
            
            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.style.borderColor = '#667eea';
                uploadArea.style.background = '';
                
                if (e.dataTransfer.files.length) {
                    fileInput.files = e.dataTransfer.files;
                    handleFileSelect();
                }
            });
            
            fileInput.addEventListener('change', handleFileSelect);
            
            function handleFileSelect() {
                if (fileInput.files.length) {
                    currentFile = fileInput.files[0];
                    uploadArea.innerHTML = `
                        <div style="color: #4CAF50; font-size: 1.2em;">
                            ✓ ${currentFile.name} (${(currentFile.size/1024).toFixed(1)} KB)
                        </div>
                        <div style="margin-top: 10px; color: #666;">
                            Click to change file
                        </div>
                    `;
                }
            }
            
            // Clear button
            document.getElementById('clear-btn').addEventListener('click', () => {
                fileInput.value = '';
                currentFile = null;
                resultDiv.style.display = 'none';
                uploadArea.innerHTML = `
                    <div class="upload-icon">📁</div>
                    <div class="upload-text">Click to upload or drag & drop</div>
                    <div class="upload-hint">Supported: JPG, PNG, JPEG (Max 5MB)</div>
                `;
            });
            
            // Predict button
            document.getElementById('predict-btn').addEventListener('click', async () => {
                if (!currentFile) {
                    showError('Please select an image file first');
                    return;
                }
                
                const formData = new FormData();
                formData.append('image', currentFile);
                formData.append('preprocessing', preprocessingSelect.value);
                
                resultDiv.innerHTML = '<div class="loading">Analyzing digit</div>';
                resultDiv.style.display = 'block';
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (result.success) {
                        displayResult(result);
                    } else {
                        showError(result.error || 'Prediction failed');
                    }
                } catch (error) {
                    showError('Network error: ' + error.message);
                }
            });
            
            function displayResult(result) {
                const confidencePercent = (result.confidence * 100).toFixed(1);
                
                // Create top predictions HTML
                let topPredictionsHTML = '';
                result.top_predictions.forEach((pred, index) => {
                    const predPercent = (pred.confidence * 100).toFixed(1);
                    const barWidth = pred.confidence * 100;
                    topPredictionsHTML += `
                        <div class="top-prediction">
                            <div class="top-digit">${pred.digit}</div>
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: ${barWidth}%"></div>
                            </div>
                            <div class="top-confidence">${predPercent}% confidence</div>
                        </div>
                    `;
                });
                
                resultDiv.innerHTML = `
                    <div class="result-header">Prediction Result</div>
                    
                    <div class="prediction-card">
                        <div style="color: #666; margin-bottom: 10px;">Predicted Digit</div>
                        <div class="prediction-digit">${result.prediction}</div>
                        
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${confidencePercent}%"></div>
                        </div>
                        
                        <div class="confidence-text">
                            ${confidencePercent}% confidence (${result.model_confidence})
                        </div>
                        
                        <div style="margin-top: 20px; color: #888; font-size: 0.9em;">
                            Processing time: ${result.inference_time_ms}ms • Mode: ${result.preprocessing_mode}
                        </div>
                    </div>
                    
                    <div style="color: #666; margin: 20px 0 10px 0;">Top 3 Predictions:</div>
                    <div class="top-predictions">
                        ${topPredictionsHTML}
                    </div>
                `;
            }
            
            function showError(message) {
                resultDiv.innerHTML = `
                    <div class="error">
                        <strong>Error:</strong> ${message}
                    </div>
                `;
                resultDiv.style.display = 'block';
            }
        </script>
    </body>
    </html>
    '''

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model": "MNISTCNN",
        "device": str(device),
        "model_loaded": loaded,
        "endpoints": ["/", "/health", "/predict", "/predict_batch", "/api/predict"]
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint."""
    request_id = str(uuid.uuid4())[:8]
    
    try:
        if 'image' not in request.files:
            return jsonify({
                "success": False,
                "error": "No image file provided",
                "request_id": request_id
            }), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({
                "success": False,
                "error": "Empty filename",
                "request_id": request_id
            }), 400
        
        # Read image bytes
        image_bytes = file.read()
        
        # Get preprocessing mode
        preprocessing_mode = request.form.get('preprocessing', 'enhanced')
        
        # Validate preprocessing mode
        if preprocessing_mode not in ['basic', 'enhanced']:
            preprocessing_mode = 'enhanced'
        
        # Make prediction
        result = predict_digit(image_bytes, preprocessing_mode, request_id)
        result['request_id'] = request_id
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "request_id": request_id
        }), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for programmatic access (JSON input)."""
    request_id = str(uuid.uuid4())[:8]
    
    try:
        if not request.is_json:
            return jsonify({
                "success": False,
                "error": "Request must be JSON",
                "request_id": request_id
            }), 400
        
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({
                "success": False,
                "error": "No image data provided",
                "request_id": request_id
            }), 400
        
        # Decode base64 image
        image_b64 = data['image']
        if 'base64,' in image_b64:
            image_b64 = image_b64.split('base64,')[1]
        
        try:
            image_bytes = base64.b64decode(image_b64)
        except:
            return jsonify({
                "success": False,
                "error": "Invalid base64 encoding",
                "request_id": request_id
            }), 400
        
        # Get preprocessing mode
        preprocessing_mode = data.get('preprocessing', 'enhanced')
        
        # Make prediction
        result = predict_digit(image_bytes, preprocessing_mode, request_id)
        result['request_id'] = request_id
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "request_id": request_id
        }), 500

# ============================================================================
# ALTERNATIVE: IF NO MODEL FOUND, TRAIN A SIMPLE ONE
# ============================================================================

def train_simple_model():
    """Train a simple model if no pre-trained model exists."""
    print("\nTraining a simple model...")
    
    # Create a simple dataset
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Use MNIST from torchvision
    train_dataset = datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=64, 
        shuffle=True
    )
    
    # Train for a few epochs
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(3):  # Just 3 epochs for quick training
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx > 50:  # Limit batches
                break
            
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}/3 completed")
    
    model.eval()
    
    # Save the model
    torch.save(model.state_dict(), 'simple_mnist_model.pth')
    print("Simple model saved as 'simple_mnist_model.pth'")
    
    return model

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    # Configuration
    app.config['DEBUG'] = True
    app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024
    
    print("=" * 60)
    print("MNIST Digit Recognition API")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    if not loaded:
        print("\n⚠  No pre-trained model found.")
        response = input("Do you want to train a simple model? (y/n): ")
        if response.lower() == 'y':
            train_simple_model()
            # Try loading the simple model
            try:
                model.load_state_dict(torch.load('simple_mnist_model.pth', map_location=device))
                print("✓ Simple model loaded")
                loaded = True
            except:
                print("✗ Failed to load simple model")
    
    print("\nEndpoints:")
    print("  • GET  /              - Web interface")
    print("  • GET  /health        - Health check")
    print("  • POST /predict       - Single image prediction")
    print("  • POST /api/predict   - JSON API")
    print("\nStarting server on http://127.0.0.1:5000")
    print("=" * 60)
    
    # Run Flask app
    app.run(host='127.0.0.1', port=5000, debug=True)
