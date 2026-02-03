from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import pickle
import sys
import traceback
from layers import Conv2D, MaxPool2D, Flatten, FullyConnected

# -------------------------
# Activations
# -------------------------
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# -------------------------
# Load model
# -------------------------
with open("simple_cnn_model.pkl", "rb") as f:
    saved_model = pickle.load(f)

conv1 = saved_model["conv1"]
pool1 = saved_model["pool1"]
conv2 = saved_model["conv2"]
pool2 = saved_model["pool2"]
flatten = saved_model["flatten"]
fc = saved_model["fc"]

print("✅ Model loaded successfully")

# -------------------------
# Flask
# -------------------------
app = Flask(__name__)

def preprocess_image(file):
    img = Image.open(file).convert("L").resize((28,28))
    arr = np.array(img, dtype=np.float32)/255.0
    arr = 1.0 - arr           # invert for MNIST
    arr = arr[..., np.newaxis]
    arr = np.expand_dims(arr,0)
    return arr

def predict_digit(img_array):
    out = conv1.forward(img_array)
    out = relu(out)
    out = pool1.forward(out)
    out = conv2.forward(out)
    out = relu(out)
    out = pool2.forward(out)
    out = flatten.forward(out)
    logits = fc.forward(out)
    probs = softmax(logits)
    digit = int(np.argmax(probs,axis=1)[0])
    confidence = float(probs[0,digit])
    return digit, confidence

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message":"MNIST CNN Predictor API","endpoint":"/predict","method":"POST"})

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error":"No file uploaded"}),400
    file = request.files["file"]
    try:
        img = preprocess_image(file)
        digit, conf = predict_digit(img)
        return jsonify({"predicted_digit":digit,"confidence":conf})
    except Exception:
        traceback.print_exc()
        return jsonify({"error":"Prediction failed"}),500

def startup_test():
    print("🔍 Running startup test...")
    dummy = np.zeros((1,28,28,1),dtype=np.float32)
    digit, conf = predict_digit(dummy)
    print(f"✅ Startup test OK → digit={digit}, conf={conf:.4f}")

if __name__=="__main__":
    startup_test()
    print("🚀 Starting server at http://127.0.0.1:5000")
    app.run(debug=True, host="127.0.0.1", port=5000, use_reloader=False)
