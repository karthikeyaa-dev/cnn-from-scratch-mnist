from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import io
import scipy.ndimage as ndi

app = Flask(__name__)

# -----------------------------
# Model definition (EXACT MATCH)
# -----------------------------
class MNISTCNN(nn.Module):
    def __init__(self, dropout_rate: float = 0.3):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate / 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        return self.fc_layers(x)

# -----------------------------
# Load trained model
# -----------------------------
device = torch.device("cpu")

model = MNISTCNN(dropout_rate=0.3).to(device)
checkpoint = torch.load("mnist_cnn_model_v2.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# -----------------------------
# Image preprocessing (MNIST-style)
# -----------------------------
def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    img = np.array(img).astype(np.float32)

    # White digit on black → NO inversion
    img /= 255.0

    # --- Remove noise ---
    img = ndi.gaussian_filter(img, sigma=0.6)

    # --- Adaptive threshold ---
    thresh = max(0.2, img.mean())
    img = (img > thresh).astype(np.float32)

    # --- Thinning / normalization ---
    img = ndi.binary_erosion(img, iterations=1).astype(np.float32)

    # --- Bounding box ---
    coords = np.where(img > 0)
    if coords[0].size == 0:
        raise ValueError("No digit detected")

    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()
    digit = img[y_min:y_max + 1, x_min:x_max + 1]

    # --- Resize longest side to 20 ---
    h, w = digit.shape
    scale = 20.0 / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    digit = Image.fromarray((digit * 255).astype(np.uint8))
    digit = digit.resize((new_w, new_h), Image.BILINEAR)
    digit = np.array(digit).astype(np.float32) / 255.0

    # --- Pad to 28x28 ---
    canvas = np.zeros((28, 28), dtype=np.float32)
    y_off = (28 - new_h) // 2
    x_off = (28 - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = digit

    # --- Center of mass (CRITICAL) ---
    cy, cx = ndi.center_of_mass(canvas)
    shift_y = int(round(14 - cy))
    shift_x = int(round(14 - cx))
    canvas = ndi.shift(canvas, shift=(shift_y, shift_x), mode="constant")

    return torch.from_numpy(canvas).unsqueeze(0).unsqueeze(0)

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        img_bytes = file.read()
        img_tensor = preprocess_image(img_bytes).to(device)

        with torch.no_grad():
            logits = model(img_tensor)
            probs = F.softmax(logits, dim=1)[0]

            prediction = int(torch.argmax(probs).item())
            confidence = float(probs[prediction].item())

        return jsonify({
            "prediction": prediction,
            "confidence": confidence,
            "all_probabilities": probs.tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
