import numpy as np
import torch
import pickle
from typing import List

class Conv2D:
    """2D Convolution layer with vectorized operations."""
    
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: int, 
                 stride: int = 1, 
                 padding: int = 0):
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding



        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.kernels = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale
        self.bias = np.zeros((out_channels,))
        
        # Cache for backward pass
        self.input_cache = None
        self.padded_input_cache = None

    def load_from_pytorch(self, pytorch_conv):
        self.kernels = pytorch_conv.weight.detach().numpy()
        self.bias = pytorch_conv.bias.detach().numpy()
    
    def im2col(self, x: np.ndarray) -> np.ndarray:
        """Convert image to column format for efficient convolution."""
        N, C, H, W = x.shape
        k = self.kernel_size
        s = self.stride
        
        # Calculate output dimensions
        out_H = (H - k) // s + 1
        out_W = (W - k) // s + 1
        
        # Create column matrix
        cols = np.zeros((N, C, k, k, out_H, out_W))
        
        # Efficient slicing using broadcasting
        for i in range(k):
            i_max = i + out_H * s
            for j in range(k):
                j_max = j + out_W * s
                cols[:, :, i, j, :, :] = x[:, :, i:i_max:s, j:j_max:s]
        
        # Reshape to (N * out_H * out_W, C * k * k)
        cols = cols.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_H * out_W, -1)
        
        return cols, out_H, out_W
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass with vectorized operations.
        Input: (N, C, H, W)
        Output: (N, out_channels, out_H, out_W)
        """
        self.input_cache = x
        N, C, H, W = x.shape
        
        # Add padding
        if self.padding > 0:
            x_padded = np.pad(x, 
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
        kernels_reshaped = self.kernels.reshape(self.out_channels, -1).T
        
        # Perform convolution as matrix multiplication
        output = np.dot(cols, kernels_reshaped) + self.bias
        
        # Reshape back to image format
        output = output.reshape(N, out_H, out_W, self.out_channels).transpose(0, 3, 1, 2)
        
        return output
    
    def backward(self, dL_dout: np.ndarray, lr: float = 0.01) -> np.ndarray:
        """
        Backward pass with vectorized operations.
        """
        N, C_out, out_H, out_W = dL_dout.shape
        N, C_in, H, W = self.input_cache.shape
        
        # Convert dL_dout to column format
        dL_dout_reshaped = dL_dout.transpose(0, 2, 3, 1).reshape(-1, C_out)
        
        # Convert padded input to column format
        cols, _, _ = self.im2col(self.padded_input_cache)
        
        # Compute gradients
        dL_dkernels = np.dot(cols.T, dL_dout_reshaped).T.reshape(self.kernels.shape)
        dL_dbias = np.sum(dL_dout_reshaped, axis=0)
        
        # Compute input gradient
        kernels_reshaped = self.kernels.reshape(self.out_channels, -1)
        dL_dcols = np.dot(dL_dout_reshaped, kernels_reshaped)
        
        # Convert column gradient back to image format
        dL_dinput_padded = self.col2im(dL_dcols, H + 2 * self.padding, W + 2 * self.padding)
        
        # Remove padding from input gradient
        if self.padding > 0:
            dL_dinput = dL_dinput_padded[:, :, 
                                         self.padding:self.padding + H,
                                         self.padding:self.padding + W]
        else:
            dL_dinput = dL_dinput_padded
        
        # Update parameters
        self.kernels -= lr * dL_dkernels / N
        self.bias -= lr * dL_dbias / N
        
        return dL_dinput
    
    def col2im(self, cols: np.ndarray, H: int, W: int) -> np.ndarray:
        """Convert column format back to image format."""
        N = self.input_cache.shape[0]
        k = self.kernel_size
        s = self.stride
        
        out_H = (H - k) // s + 1
        out_W = (W - k) // s + 1
        
        # Reshape columns
        cols = cols.reshape(N, out_H, out_W, self.in_channels, k, k)
        cols = cols.transpose(0, 3, 4, 5, 1, 2)
        
        # Initialize output image
        img = np.zeros((N, self.in_channels, H, W))
        
        # Accumulate slices back to image
        for i in range(k):
            i_max = i + out_H * s
            for j in range(k):
                j_max = j + out_W * s
                img[:, :, i:i_max:s, j:j_max:s] += cols[:, :, i, j, :, :]
        
        return img

class MaxPool2D:
    """2D Max Pooling layer with vectorized operations."""
    
    def __init__(self, pool_size: int = 2, stride: int = None):
        self.pool_size = pool_size
        self.stride = stride or pool_size
        
    def forward(self, x: np.ndarray) -> np.ndarray:
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
        self.max_indices = np.argmax(x_reshaped, axis=3)
        output = np.max(x_reshaped, axis=3)
        
        # Reshape back
        output = output.reshape(N, C, out_H, out_W)
        
        # Cache for backward pass
        self.input_shape = x.shape
        self.output_shape = output.shape
        
        return output
    
    def backward(self, dL_dout: np.ndarray) -> np.ndarray:
        """Backward pass."""
        N, C, out_H, out_W = dL_dout.shape
        k = self.pool_size
        
        # Create gradient array
        dL_dinput = np.zeros((N, C, out_H * out_W, k * k))
        
        # Place gradients at max positions
        dL_dinput[np.arange(N)[:, None, None], 
                 np.arange(C)[None, :, None], 
                 np.arange(out_H * out_W)[None, None, :], 
                 self.max_indices] = dL_dout.reshape(N, C, -1)
        
        # Reshape back to original dimensions
        dL_dinput = dL_dinput.reshape(N, C, out_H, out_W, k, k)
        dL_dinput = dL_dinput.transpose(0, 1, 2, 4, 3, 5)
        dL_dinput = dL_dinput.reshape(self.input_shape)
        
        return dL_dinput


class Flatten:
    """Flatten layer."""
    
    def __init__(self):
        self.input_shape = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Flatten input."""
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)
    
    def backward(self, dL_dout: np.ndarray) -> np.ndarray:
        """Reshape gradient to input shape."""
        return dL_dout.reshape(self.input_shape)


class FullyConnected:
    """Fully connected (dense) layer."""
    
    def __init__(self, input_size: int, output_size: int):
        # Xavier initialization
        scale = np.sqrt(2.0 / (input_size + output_size))
        self.weights = np.random.randn(input_size, output_size) * scale
        self.biases = np.zeros(output_size)
        
        self.input_cache = None

    def load_from_pytorch(self, pytorch_linear):
        self.weights = pytorch_linear.weight.detach().numpy().T
        self.biases = pytorch_linear.bias.detach().numpy()
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        self.input_cache = x
        return np.dot(x, self.weights) + self.biases
    
    def backward(self, dL_dout: np.ndarray, lr: float = 0.01) -> np.ndarray:
        """Backward pass."""
        N = self.input_cache.shape[0]
        
        # Gradients
        dL_dW = np.dot(self.input_cache.T, dL_dout)
        dL_db = np.sum(dL_dout, axis=0)
        dL_dx = np.dot(dL_dout, self.weights.T)
        
        # Update parameters
        self.weights -= lr * dL_dW / N
        self.biases -= lr * dL_db / N
        
        return dL_dx


class ReLU:
    """ReLU activation layer."""
    
    def __init__(self):
        self.input_cache = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        self.input_cache = x
        return np.maximum(0, x)
    
    def backward(self, dL_dout: np.ndarray) -> np.ndarray:
        """Backward pass."""
        return dL_dout * (self.input_cache > 0)

class Model:
    """Neural network model."""
    
    def __init__(self, layers: List):
        self.layers = layers
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through all layers."""
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, dL_dout: np.ndarray, lr: float = 0.01) -> np.ndarray:
        """Backward pass through all layers."""
        for layer in reversed(self.layers):
            if hasattr(layer, 'backward'):
                if isinstance(layer, (FullyConnected, Conv2D)):
                    dL_dout = layer.backward(dL_dout, lr)
                else:
                    dL_dout = layer.backward(dL_dout)
        return dL_dout
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Make predictions."""
        logits = self.forward(x)
        return np.argmax(logits, axis=1)
    
    def save(self, path: str):
        """Save model to file."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Model saved to {path}")
    
    @staticmethod
    def load(path: str) -> 'Model':
        """Load model from file."""
        with open(path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from {path}")
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







