import numpy as np
import torch
import pickle
from typing import List
import jax
import jax.numpy as jnp

class Conv2D:
    """2D Convolution layer with vectorized operations using JAX."""

    def __init__(
        self,
        key: jax.random.PRNGKey,   # JAX requires a key for randomness
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # He initialization scale
        scale = jnp.sqrt(2.0 / (in_channels * kernel_size * kernel_size))

        # Split the key for kernel initialization
        key, subkey = jax.random.split(key)
        self.kernels = jax.random.normal(
            subkey, 
            shape=(out_channels, in_channels, kernel_size, kernel_size)
        ) * scale

        # Bias initialized to zeros
        self.bias = jnp.zeros((out_channels,))

        # Cache for backward pass
        self.input_cache: Optional[jnp.ndarray] = None
        self.padded_input_cache: Optional[jnp.ndarray] = None

    def load_from_pytorch(self, pytorch_conv):
        self.kernels = pytorch_conv.weight.detach().numpy()
        self.bias = pytorch_conv.bias.detach().numpy()
    
    def im2col(self, x: jnp.ndarray):
        """Vectorized im2col for 2D convolution."""
        N, C, H, W = x.shape
        k = self.kernel_size
        s = self.stride

        # Calculate output dimensions
        out_H = (H - k) // s + 1
        out_W = (W - k) // s + 1

        # Compute row indices
        i0 = jnp.arange(k)[:, None] + jnp.arange(out_H) * s  # shape (k, out_H)
        i_idx = jnp.repeat(i0, k, axis=0).reshape(-1)        # flatten for all patches

        # Compute column indices
        j0 = jnp.arange(k)[:, None] + jnp.arange(out_W) * s
        j_idx = jnp.tile(j0, (k, 1)).reshape(-1)

        # Broadcast over channels and batch
        x_cols = x[:, :, i_idx[:, None], j_idx[None, :]]  # shape (N, C, k*k*out_H*out_W)
        
        # Reshape to (N*out_H*out_W, C*k*k)
        cols = x_cols.transpose(0, 3, 1, 2).reshape(N*out_H*out_W, -1)
        return cols, out_H, out_W
    

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass with vectorized operations (JAX).
        Input: (N, C, H, W)
        Output: (N, out_channels, out_H, out_W)
        """
        self.input_cache = x
        N, C, H, W = x.shape

        # Add padding
        if self.padding > 0:
            x_padded = jnp.pad(x, 
                            ((0, 0), (0, 0), 
                                (self.padding, self.padding), 
                                (self.padding, self.padding)), 
                            mode='constant')
            H_padded = H + 2 * self.padding
            W_padded = W + 2 * self.padding
        else:
            x_padded = x
            H_padded, W_padded = H, W

        self.padded_input_cache = x_padded

        # Convert to column format
        cols, out_H, out_W = self.im2col(x_padded)

        # Reshape kernels for matrix multiplication
        kernels_reshaped = self.kernels.reshape(self.out_channels, -1).T  # shape (C*k*k, out_channels)

        # Perform convolution as matrix multiplication
        output = cols @ kernels_reshaped + self.bias  # (N*out_H*out_W, out_channels)

        # Reshape back to image format
        output = output.reshape(N, out_H, out_W, self.out_channels).transpose(0, 3, 1, 2)

        return output

    

    def backward(self, dL_dout: jnp.ndarray, lr: float = 0.01) -> jnp.ndarray:
        """
        Backward pass with vectorized operations (JAX).
        """
        N, C_out, out_H, out_W = dL_dout.shape
        N, C_in, H, W = self.input_cache.shape

        # Reshape dL_dout to (N*out_H*out_W, C_out)
        dL_dout_reshaped = dL_dout.transpose(0, 2, 3, 1).reshape(-1, C_out)

        # Convert padded input to column format
        cols, _, _ = self.im2col(self.padded_input_cache)  # shape (N*out_H*out_W, C_in*k*k)

        # Gradients w.r.t kernels and bias
        dL_dkernels = (dL_dout_reshaped.T @ cols).reshape(self.kernels.shape)  # (out_channels, C_in, k, k)
        dL_dbias = jnp.sum(dL_dout_reshaped, axis=0)

        # Gradient w.r.t input columns
        kernels_reshaped = self.kernels.reshape(self.out_channels, -1)  # (out_channels, C_in*k*k)
        dL_dcols = dL_dout_reshaped @ kernels_reshaped  # (N*out_H*out_W, C_in*k*k)

        # Convert column gradients back to image format
        dL_dinput_padded = self.col2im(dL_dcols, H + 2 * self.padding, W + 2 * self.padding)

        # Remove padding from input gradient
        if self.padding > 0:
            dL_dinput = dL_dinput_padded[:, :, 
                                        self.padding:self.padding + H,
                                        self.padding:self.padding + W]
        else:
            dL_dinput = dL_dinput_padded

        # Update parameters (in-place for simplicity)
        self.kernels -= lr * dL_dkernels / N
        self.bias -= lr * dL_dbias / N

        return dL_dinpu_ 
 
    def col2im(self, cols: jnp.ndarray, H: int, W: int) -> jnp.ndarray:
        """
        Convert column format back to image format (JAX).
        """
        N = self.input_cache.shape[0]
        k = self.kernel_size
        s = self.stride
        C = self.in_channels

        out_H = (H - k) // s + 1
        out_W = (W - k) // s + 1

        # Reshape columns
        cols_reshaped = cols.reshape(N, out_H, out_W, C, k, k).transpose(0, 3, 4, 5, 1, 2)
        # shape: (N, C, k, k, out_H, out_W)

        # Initialize output image
        img = jnp.zeros((N, C, H, W))

        # Compute row and column indices for broadcasting
        i0 = jnp.arange(k)[:, None] + jnp.arange(out_H) * s  # shape (k, out_H)
        j0 = jnp.arange(k)[:, None] + jnp.arange(out_W) * s  # shape (k, out_W)
        i_idx = i0.reshape(-1)
        j_idx = j0.reshape(-1)

        # Flatten kernel dimensions
        cols_flat = cols_reshaped.reshape(N, C, k*k, out_H*out_W)

        # Scatter-add all values back into image
        for n in range(N):
            for c in range(C):
                # Use advanced indexing and sum into positions
                img = img.at[n, c, i_idx[:, None], j_idx[None, :]].add(cols_flat[n, c])

        return img


class MaxPool2D:
    """2D Max Pooling layer with vectorized operations (JAX)."""
    
    def __init__(self, pool_size: int = 2, stride: int = None):
        self.pool_size = pool_size
        self.stride = stride or pool_size
        
    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass."""
        N, C, H, W = x.shape
        k = self.pool_size
        s = self.stride

        out_H = H // s
        out_W = W // s

        # Reshape to extract pooling regions
        x_reshaped = x.reshape(N, C, out_H, k, out_W, k)
        x_reshaped = x_reshaped.transpose(0, 1, 2, 4, 3, 5).reshape(N, C, out_H * out_W, k * k)

        # Find max in each region
        self.max_indices = jnp.argmax(x_reshaped, axis=3)
        output = jnp.max(x_reshaped, axis=3)

        # Reshape back to (N, C, out_H, out_W)
        output = output.reshape(N, C, out_H, out_W)

        # Cache shapes for backward
        self.input_shape = x.shape
        self.output_shape = output.shape

        return output
    
    def backward(self, dL_dout: jnp.ndarray) -> jnp.ndarray:
        """Backward pass."""
        N, C, out_H, out_W = dL_dout.shape
        k = self.pool_size

        # Flatten output gradient
        dL_dout_flat = dL_dout.reshape(N, C, out_H * out_W)

        # Create gradient array for flattened pooling regions
        dL_dinput_flat = jnp.zeros((N, C, out_H * out_W, k * k))

        # Use JAX scatter to place gradients at max indices
        dL_dinput_flat = dL_dinput_flat.at[
            jnp.arange(N)[:, None, None],
            jnp.arange(C)[None, :, None],
            jnp.arange(out_H * out_W)[None, None, :],
            self.max_indices
        ].set(dL_dout_flat)

        # Reshape back to original input shape
        dL_dinput = dL_dinput_flat.reshape(N, C, out_H, out_W, k, k)
        dL_dinput = dL_dinput.transpose(0, 1, 2, 4, 3, 5)
        dL_dinput = dL_dinput.reshape(self.input_shape)

        return dL_dinput

class Flatten:
    """Flatten layer (JAX)."""
    
    def __init__(self):
        self.input_shape = None
    
    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """Flatten input."""
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)
    
    def backward(self, dL_dout: jnp.ndarray) -> jnp.ndarray:
        """Reshape gradient to input shape."""
        return dL_dout.reshape(self.input_shape)


class FullyConnected:
    """Fully connected (dense) layer (JAX)."""
    
    def __init__(self, input_size: int, output_size: int, key: jax.random.KeyArray):
        # Xavier initialization
        k1, k2 = jax.random.split(key)
        scale = jnp.sqrt(2.0 / (input_size + output_size))
        self.weights = jax.random.normal(k1, (input_size, output_size)) * scale
        self.biases = jnp.zeros(output_size)
        self.input_cache = None

    def load_from_pytorch(self, pytorch_linear):
        self.weights = pytorch_linear.weight.detach().numpy().T
        self.biases = pytorch_linear.bias.detach().numpy()
    
    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass."""
        self.input_cache = x
        return x @ self.weights + self.biases
    
    def backward(self, dL_dout: jnp.ndarray, lr: float = 0.01) -> jnp.ndarray:
        """Backward pass."""
        N = self.input_cache.shape[0]

        # Gradients
        dL_dW = self.input_cache.T @ dL_dout
        dL_db = jnp.sum(dL_dout, axis=0)
        dL_dx = dL_dout @ self.weights.T

        # Update parameters
        self.weights -= lr * dL_dW / N
        self.biases -= lr * dL_db / N

        return dL_dx

class ReLU:
    """ReLU activation layer (JAX)."""
    
    def __init__(self):
        self.input_cache = None
    
    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass."""
        self.input_cache = x
        return jnp.maximum(0, x)
    
    def backward(self, dL_dout: jnp.ndarray) -> jnp.ndarray:
        """Backward pass."""
        return dL_dout * (self.input_cache > 0)

class Model:
    """Neural network model (JAX)."""
    
    def __init__(self, layers: List):
        self.layers = layers
    
    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through all layers."""
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, dL_dout: jnp.ndarray, lr: float = 0.01) -> jnp.ndarray:
        """Backward pass through all layers."""
        for layer in reversed(self.layers):
            if hasattr(layer, 'backward'):
                # Only layers with learnable params take lr
                if isinstance(layer, (FullyConnected, Conv2D)):
                    dL_dout = layer.backward(dL_dout, lr)
                else:
                    dL_dout = layer.backward(dL_dout)
        return dL_dout
    
    def predict(self, x: jnp.ndarray) -> jnp.ndarray:
        """Make predictions (argmax logits)."""
        logits = self.forward(x)
        return jnp.argmax(logits, axis=1)
    
    def save(self, path: str):
        """Save model to file."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {path}")
    
    @staticmethod
    def load(path: str) -> 'Model':
        """Load model from file."""
        with open(path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {path}")
        return model

    def load_pytorch_weights(self, pytorch_model_path):
        checkpoint = torch.load(pytorch_model_path, map_location='cpu')
        state_dict = checkpoint['model_state_dict']
        
        print(f"Model has {len(self.layers)} layers")
        for i, layer in enumerate(self.layers):
            print(f"  {i}: {layer.__class__.__name__}")
        
        print("\nLoading weights...")
        
        # Simple mapping for our simpler model
        try:
            # Layer 0: First conv (32 filters)
            self.layers[0].kernels = state_dict['conv_layers.0.weight'].numpy()
            self.layers[0].bias = state_dict['conv_layers.0.bias'].numpy()
            print("Loaded conv layer 0")
            
            # Layer 3: Second conv (64 filters) - using conv_layers.8 from PyTorch
            self.layers[3].kernels = state_dict['conv_layers.8.weight'].numpy()
            self.layers[3].bias = state_dict['conv_layers.8.bias'].numpy()
            print("Loaded conv layer 3")
            
            # Layer 6: First FC (256 units)
            self.layers[6].weights = state_dict['fc_layers.1.weight'].numpy().T
            self.layers[6].biases = state_dict['fc_layers.1.bias'].numpy()
            print("Loaded FC layer 6")
            
            # Layer 8: Last FC (10 units)
            self.layers[8].weights = state_dict['fc_layers.9.weight'].numpy().T
            self.layers[8].biases = state_dict['fc_layers.9.bias'].numpy()
            print("Loaded FC layer 8")
            
        except (KeyError, IndexError) as e:
            print(f"Error loading weights: {e}")
            print("Available keys:", list(state_dict.keys()))
            raise
        
        print("Weights loaded successfully!")

# Define a SIMPLER CNN that actually matches weights we have
model = Model([
    # Just use the FIRST conv layer from PyTorch
    Conv2D(in_channels=1, out_channels=32, kernel_size=3, padding=1),
    ReLU(),
    MaxPool2D(pool_size=2, stride=2),
    
    # Just use ONE more conv layer
    Conv2D(in_channels=32, out_channels=64, kernel_size=3, padding=1),
    ReLU(),
    MaxPool2D(pool_size=2, stride=2),
    
    Flatten(),
    
    # Use the FIRST and LAST FC layers
    FullyConnected(64 * 7 * 7, 256),  # 64 channels, after 2 max pools: 28→14→7
    ReLU(),
    FullyConnected(256, 10)
])
