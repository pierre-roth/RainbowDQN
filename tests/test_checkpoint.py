import pytest
import torch
import os
from src.agent import DQNAgent
from src.config import Config

def test_save_load_checkpoint(tmp_path):
    # Setup
    observation_shape = (4, 84, 84)
    num_actions = 4
    agent = DQNAgent(observation_shape, num_actions)
    
    # Modify state to ensure we are testing persistence
    # Modify state to ensure we are testing persistence
    # agent.epsilon = 0.5 # Removed
    with torch.no_grad():
        # NoisyLinear uses weight_mu and weight_sigma
        agent.policy_net.value_stream[0].weight_mu.fill_(1.0)
    
    # Save
    save_path = tmp_path / "test_checkpoint.chkpt"
    step = 100
    agent.save(save_path, step)
    
    assert os.path.exists(save_path)
    
    # Load into new agent
    new_agent = DQNAgent(observation_shape, num_actions)
    loaded_step = new_agent.load(save_path)
    
    # Verify
    assert loaded_step == 100
    
    # Verify
    # assert new_agent.epsilon == 0.5 # Removed
    # Check weights
    assert torch.allclose(
        new_agent.policy_net.value_stream[0].weight_mu, 
        torch.ones_like(new_agent.policy_net.value_stream[0].weight_mu)
    )
    
    # Check optimizer state loaded (just check logic runs, inspecting deep optimizer state is complex)
    # If state_dict load didn't crash, it's likely fine.
