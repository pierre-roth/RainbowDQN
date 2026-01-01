import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from pathlib import Path
from .model import DQN
from .config import Config

class DQNAgent:
    def __init__(self, observation_shape: tuple, num_actions: int, device: str = Config.DEVICE):
        """
        Initialize the DQN Agent with Noisy Networks and N-Step parameters.

        Args:
            observation_shape (tuple): Shape of the observation (Channels, Height, Width).
            num_actions (int): Number of discrete actions available.
            device (str): Device to run the agent on ('cpu' or 'cuda'/'mps').
        """
        self.device = device
        self.num_actions = num_actions
        
        # Initialize Networks
        self.policy_net = DQN(observation_shape, num_actions).to(device)
        self.target_net = DQN(observation_shape, num_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=Config.LEARNING_RATE, eps=1.5e-4)
        
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select an action. Exploration is inherent in the Noisy Networks.
        
        Args:
            state (np.ndarray): Current state/observation.
            training (bool): If True, use noisy weights. If False, use mean weights.

        Returns:
            int: Selected action index.
        """
        # Ensure correct mode
        self.policy_net.train(training)
        
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device) / 255.0
                if state_t.ndim == 3:
                     state_t = state_t.unsqueeze(0)
            else:
                 state_t = state
            
            # Reset noise for correct exploration behavior if training
            if training:
                self.policy_net.reset_noise()
            
            q_values = self.policy_net(state_t)
            return q_values.argmax(dim=1).item()
            

        
    def learn(self, batch: tuple, weights: torch.Tensor = None) -> tuple[float, float, np.ndarray]:
        """
        Update the policy network using a batch of experiences.

        Args:
            batch (tuple): Tuple containing (obs, acts, rews, next_obs, dones).
            weights (torch.Tensor, optional): Importance sampling weights for PER.

        Returns:
            tuple: (loss, mean_q_value, td_errors)
        """
        obs, acts, rews, next_obs, dones = batch
        
        # Double DQN Logic + N-Step Returns
        
        self.policy_net.train()
        self.target_net.eval()
        
        # Resample noise for learning
        self.policy_net.reset_noise()
        self.target_net.reset_noise()
        
        with torch.no_grad():
            # Action selection from online network
            next_state_actions = self.policy_net(next_obs).argmax(dim=1, keepdim=True)
            # Evaluation from target network
            next_q_values = self.target_net(next_obs).gather(1, next_state_actions).squeeze(1)
            
            # Config.GAMMA ** Config.N_STEP for N-step returns
            # Replay buffer stores sum of n-step rewards in 'rews'
            discount = Config.GAMMA ** Config.N_STEP
            expected_q_values = rews + (discount * next_q_values * (1 - dones))
            
        current_q_values = self.policy_net(obs).gather(1, acts.unsqueeze(1)).squeeze(1)
        
        # Loss Calculation
        loss_elementwise = F.smooth_l1_loss(current_q_values, expected_q_values, reduction='none')
        
        if weights is not None:
            loss = (loss_elementwise * weights).mean()
        else:
            loss = loss_elementwise.mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()
        
        with torch.no_grad():
            td_errors = torch.abs(current_q_values - expected_q_values).cpu().numpy()
            
        return loss.item(), current_q_values.mean().item(), td_errors
        
    def sync_target_network(self):
        """
        Synchronize target network with policy network.
        Strict Hard Update.
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def save(self, path: Path | str, step: int) -> None:
        """
        Save the agent's state, including model weights, optimizer, and current step.
        """
        torch.save({
            'model_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': step
        }, path)
        
    def load(self, path: Path | str) -> int:
        """
        Load agent state. Returns the step count loaded (or 0 if not found).
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['model_state_dict'])
        self.target_net.load_state_dict(self.policy_net.state_dict())
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint.get('step', 0)
