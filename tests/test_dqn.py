import pytest
import torch
import numpy as np
from src.config import Config
from src.wrappers import make_env
from src.agent import DQNAgent
from src.buffer import PrioritizedReplayBuffer

def test_mps_availability():
    """Verify that MPS (Metal Performance Shaders) is available on this machine."""
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    assert torch.backends.mps.is_available()

def test_environment_creation():
    """Verify environment creation and observation shape."""
    env = make_env(Config.ENV_ID)
    try:
        assert env.observation_space.shape == (4, 84, 84)
        obs, _ = env.reset()
        assert obs.shape == (4, 84, 84)
    finally:
        env.close()

def test_agent_initialization():
    """Verify agent network initialization."""
    env = make_env(Config.ENV_ID)
    try:
        agent = DQNAgent(env.observation_space.shape, env.action_space.n)
        assert agent.policy_net is not None
        assert agent.target_net is not None
        
        # Check device
        expected_device = torch.device(Config.DEVICE)
        assert next(agent.policy_net.parameters()).device.type == expected_device.type
    finally:
        env.close()

def test_training_step():
    """Verify that a training step (forward + backward) runs without error."""
    env = make_env(Config.ENV_ID)
    agent = DQNAgent(env.observation_space.shape, env.action_space.n)
    buffer = PrioritizedReplayBuffer(1000, env.observation_space.shape)
    
    obs, _ = env.reset()
    
    # Fill buffer slightly more than batch size
    for _ in range(Config.BATCH_SIZE + 2):
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.add(obs, action, reward, next_obs, done)
        if done:
            obs, _ = env.reset()
        else:
            obs = next_obs
            
    # Test Learn with PER
    batch, weights, indices = buffer.sample(Config.BATCH_SIZE)
    loss, q, td_errors = agent.learn(batch, weights)
    
    assert loss is not None
    assert isinstance(loss, float)
    assert q is not None
    assert isinstance(q, float)
    assert td_errors is not None
    assert len(td_errors) == Config.BATCH_SIZE
    
    # Verify priority update
    buffer.update_priorities(indices, td_errors)
    
    env.close()
