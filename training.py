import numpy as np
from layers import Conv2D, MaxPool2D, Flatten, FullyConnected
from data import LazyMNISTDataset, MNISTDataLoader

train_dataset = LazyMNISTDataset("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz")
train_loader = MNISTDataLoader(train_dataset, batch_size=64, shuffle=True)

# Cross-entropy loss for logits
def cross_entropy_loss(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """
    logits: (N, num_classes)
    labels: (N,) integer labels
    """
    # Convert to one-hot
    num_classes = logits.shape[1]
    one_hot = jax.nn.one_hot(labels, num_classes)
    
    # Softmax + log
    log_probs = jax.nn.log_softmax(logits)
    loss = -jnp.sum(one_hot * log_probs, axis=1)  # per-sample loss
    return jnp.mean(loss)  # average over batch

# Accuracy
def accuracy(logits: jnp.ndarray, labels: jnp.ndarray) -> float:
    preds = jnp.argmax(logits, axis=1)
    return jnp.mean(preds == labels)

# Training function
def train(model: Model, train_loader, epochs=5, lr=0.01):
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        n_batches = 0
        
        for X_batch, Y_batch in train_loader:
            # Forward pass
            logits = model.forward(X_batch)
            
            # Compute loss
            loss = cross_entropy_loss(logits, Y_batch)
            
            # Compute gradient w.r.t output
            N = X_batch.shape[0]
            num_classes = logits.shape[1]
            one_hot = jax.nn.one_hot(Y_batch, num_classes)
            dL_dout = (jax.nn.softmax(logits) - one_hot) / N  # gradient of loss
            
            # Backward pass (updates weights inside layers)
            model.backward(dL_dout, lr=lr)
            
            # Metrics
            acc = accuracy(logits, Y_batch)
            epoch_loss += loss
            epoch_acc += acc
            n_batches += 1
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/n_batches:.4f} | Accuracy: {epoch_acc/n_batches:.4f}")))
