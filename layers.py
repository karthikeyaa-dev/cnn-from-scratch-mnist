import numpy as np

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.kernels = np.random.randn(
            out_channels, in_channels, kernel_size, kernel_size
        )
        self.bias = np.zeros(out_channels)

    def forward(self, input_images):
        # input_images: (N, H, W, C)
        N, H, W, C = input_images.shape
        assert C == self.in_channels

        # Padding
        if self.padding > 0:
            input_images = np.pad(
                input_images,
                ((0, 0),
                 (self.padding, self.padding),
                 (self.padding, self.padding),
                 (0, 0)),
                mode="constant"
            )

        k = self.kernel_size
        out_H = (H + 2*self.padding - k) // self.stride + 1
        out_W = (W + 2*self.padding - k) // self.stride + 1

        output = np.zeros((N, out_H, out_W, self.out_channels))

        for n in range(N):                     # batch
            for f in range(self.out_channels): # filters
                for i in range(out_H):
                    for j in range(out_W):
                        region = input_images[
                            n,
                            i*self.stride:i*self.stride+k,
                            j*self.stride:j*self.stride+k,
                            :
                        ]
                        output[n, i, j, f] = (
                            np.sum(region * self.kernels[f]) + self.bias[f]
                        )

        return output

class MaxPool2D:
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, input_images):
        # input_images: (N, H, W, C)
        N, H, W, C = input_images.shape
        k = self.pool_size
        s = self.stride

        out_H = (H - k) // s + 1
        out_W = (W - k) // s + 1

        output = np.zeros((N, out_H, out_W, C))

        for n in range(N):          # batch
            for c in range(C):      # channels
                for i in range(out_H):
                    for j in range(out_W):
                        region = input_images[
                            n,
                            i*s:i*s+k,
                            j*s:j*s+k,
                            c
                        ]
                        output[n, i, j, c] = np.max(region)

        return output


class Flatten:
    def forward(self, input_images):
        N = input_images.shape[0]
        return input_images.reshape(N, -1)


class FullyConnected:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros(output_size)

    def forward(self, x):
        # x: (N, input_size)
        self.input = x
        return np.dot(x, self.weights) + self.biases

    def backward(self, dL_dout, lr=0.01):
        # dL_dout: (N, output_size)
        dL_dW = np.dot(self.input.T, dL_dout)
        dL_db = np.sum(dL_dout, axis=0)
        dL_dx = np.dot(dL_dout, self.weights.T)

        self.weights -= lr * dL_dW
        self.biases -= lr * dL_db

        return dL_dx

class ReLU:
    def forward(self, x):
        return np.maximum(0, x)

# Define the CNN
model = Model([
    Conv2D(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1),  # 28x28x1 -> 28x28x8
    ReLU(),
    MaxPool2D(pool_size=2, stride=2),                                           # 28x28x8 -> 14x14x8
    Conv2D(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1), # 14x14x8 -> 14x14x16
    ReLU(),
    MaxPool2D(pool_size=2, stride=2),                                           # 14x14x16 -> 7x7x16
    Flatten(),                                                                   # 7*7*16 = 784
    FullyConnected(7*7*16, 128),
    ReLU(),
    FullyConnected(128, 10)  # 10 classes
])







