import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from .config import Config

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=Config.STD_INIT):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()
        
    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        
    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())
        
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
        
    def forward(self, input):
        if self.training:
            return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon,
                            self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)

class DQN(nn.Module):
    def __init__(self, input_shape: tuple, num_actions: int):
        """
        Initialize the DQN model with Dueling architecture and Noisy Nets.

        Args:
            input_shape (tuple): Shape of the input (Channels, Height, Width).
            num_actions (int): Number of possible actions.
        """
        super(DQN, self).__init__()
        
        # Input shape is (Frames, H, W) e.g., (4, 84, 84)
        c, h, w = input_shape
        
        self.features = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Compute size of flattened features
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1
            
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64
        
        # Dueling Streams with Noisy Linear Layers
        
        # Value Stream: Outputs single scalar V(s)
        self.value_stream = nn.Sequential(
            NoisyLinear(linear_input_size, 512),
            nn.ReLU(),
            NoisyLinear(512, 1)
        )
        
        # Advantage Stream: Outputs vector A(s, a)
        self.advantage_stream = nn.Sequential(
            NoisyLinear(linear_input_size, 512),
            nn.ReLU(),
            NoisyLinear(512, num_actions)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input tensor (Batch, Frames, H, W)
        Returns:
            torch.Tensor: Q-values (Batch, Num_Actions)
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        # Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values
        
    def reset_noise(self):
        """Reset all noisy layers."""
        for name, module in self.named_children():
            if 'stream' in name: # value_stream and advantage_stream
                for layer in module:
                    if isinstance(layer, NoisyLinear):
                        layer.reset_noise()
