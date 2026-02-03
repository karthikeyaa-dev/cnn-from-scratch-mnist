import numpy as np

from layers import Conv2D, MaxPool2D, Flatten, FullyConnected
from data import load_mnist_images, load_mnist_labels

# -------------------------
# Utility functions
# -------------------------
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy(pred, labels):
    N = labels.shape[0]
    return -np.sum(np.log(pred[np.arange(N), labels] + 1e-9)) / N

def accuracy(pred, labels):
    return np.mean(np.argmax(pred, axis=1) == labels)

# -------------------------
# Load MNIST
# -------------------------
train_images = load_mnist_images("Data/train-images.idx3-ubyte")
train_labels = load_mnist_labels("Data/train-labels.idx1-ubyte")

test_images = load_mnist_images("Data/t10k-images.idx3-ubyte")
test_labels = load_mnist_labels("Data/t10k-labels.idx1-ubyte")

# Normalize
train_images = train_images / 255.0
test_images = test_images / 255.0

# Add channel dimension
train_images = train_images[..., np.newaxis]  # (N, 28, 28, 1)
test_images = test_images[..., np.newaxis]

# -------------------------
# Model
# -------------------------
conv = Conv2D(in_channels=1, out_channels=8, kernel_size=3, padding=1)
pool = MaxPool2D(pool_size=2, stride=2)
flatten = Flatten()
fc = FullyConnected(14 * 14 * 8, 10)

# -------------------------
# Training
# -------------------------
epochs = 1
batch_size = 32
lr = 0.01

num_samples = train_images.shape[0]

for epoch in range(epochs):
    indices = np.random.permutation(num_samples)
    train_images = train_images[indices]
    train_labels = train_labels[indices]

    epoch_loss = 0
    epoch_acc = 0
    batches = 0

    for i in range(0, num_samples, batch_size):
        x = train_images[i:i+batch_size]
        y = train_labels[i:i+batch_size]

        # Forward pass
        out = conv.forward(x)
        out = relu(out)
        out = pool.forward(out)
        out = flatten.forward(out)
        logits = fc.forward(out)
        probs = softmax(logits)

        # Loss + accuracy
        loss = cross_entropy(probs, y)
        acc = accuracy(probs, y)

        epoch_loss += loss
        epoch_acc += acc
        batches += 1

        # Backprop (FC only)
        grad = probs
        grad[np.arange(len(y)), y] -= 1
        grad /= len(y)

        fc.backward(grad, lr)

    print(
        f"Epoch {epoch+1} | "
        f"Loss: {epoch_loss/batches:.4f} | "
        f"Accuracy: {epoch_acc/batches:.4f}"
    )

# -------------------------
# Testing
# -------------------------
out = conv.forward(test_images[:1000])
out = relu(out)
out = pool.forward(out)
out = flatten.forward(out)
logits = fc.forward(out)
probs = softmax(logits)

print("Test accuracy:", accuracy(probs, test_labels[:1000]))
