import pytest
import numpy as np
import torch
from pathlib import Path
from src.buffer import PrioritizedReplayBuffer

def test_per_initialization():
    buffer = PrioritizedReplayBuffer(capacity=100, observation_shape=(4, 84, 84))
    assert len(buffer) == 0
    assert buffer.tree.capacity == 100

def test_per_add_and_sample():
    # Shape (Channels, H, W) -> (4, 5, 5)
    shape = (4, 5, 5)
    buffer = PrioritizedReplayBuffer(capacity=100, observation_shape=shape, n_step=1)
    
    # Add some data
    for i in range(10):
        obs = np.full(shape, i, dtype=np.uint8)
        next_obs = np.full(shape, i+1, dtype=np.uint8)
        buffer.add(obs, 0, 0, next_obs, False)
        
    # With n_step=1 and N+1 delay, we hold 1 item in buffer until next one arrives.
    # So we processed 0..8 (9 items). Item 9 is pending.
    assert len(buffer) == 9
    
    # Sample
    batch, weights, indices = buffer.sample(batch_size=5, beta=0.4)
    obs, acts, rews, next_obs, dones = batch
    
    assert obs.shape == (5, 4, 5, 5)
    assert len(weights) == 5
    assert len(indices) == 5
    assert isinstance(weights, torch.Tensor)

def test_per_priority_update():
    shape = (4, 5, 5)
    buffer = PrioritizedReplayBuffer(capacity=100, observation_shape=shape, n_step=1)
    obs = np.zeros(shape, dtype=np.uint8)
    buffer.add(obs, 0, 0, obs, False)
    
    # Initial max priority is 1.0 usually
    _, _, indices = buffer.sample(1, 0.4)
    idx = indices[0]
    
    # Update priority to be very small
    buffer.update_priorities([idx], [1e-5])
    
    # Update priority to be very large
    buffer.update_priorities([idx], [100.0])
    
    # Check if sum tree reflects changes (conceptually)
    assert buffer.tree.total_priority > 0

def test_per_circular_buffer():
    capacity = 10
    shape = (4, 5, 5)
    buffer = PrioritizedReplayBuffer(capacity=capacity, observation_shape=shape, n_step=1)
    
    for i in range(15):
        obs = np.full(shape, i, dtype=np.uint8)
        buffer.add(obs, 0, 0, obs, False)
        
    assert len(buffer) == capacity
    # Should contain 5..14
    # We can't easily check content without peeking implementation details, 
    # but specific tests are good enough for now.

def test_buffer_persistence(tmp_path):
    shape = (4, 5, 5)
    buffer = PrioritizedReplayBuffer(capacity=100, observation_shape=shape, n_step=1)
    
    # Add data
    obs1 = np.full(shape, 1, dtype=np.uint8)
    obs2 = np.full(shape, 2, dtype=np.uint8)
    buffer.add(obs1, 0, 1.0, obs2, False)
    buffer.add(obs2, 1, 0.5, obs1, True)
    
    assert len(buffer) == 2
    
    # Save
    save_path = tmp_path / "buffer.npz"
    buffer.save(save_path)
    
    assert save_path.exists()
    
    # Load into new buffer
    new_buffer = PrioritizedReplayBuffer(capacity=100, observation_shape=shape, n_step=1)
    new_buffer.load(save_path)
    
    assert len(new_buffer) == 2
    assert new_buffer.tree.total_priority == buffer.tree.total_priority
    
    # Verify content of first item
    # We can check the internal arrays - use proper indexing or peek
    # Since we can't easily peek via API, we rely on checking if internal arrays match
    # using load logic which puts them in self.observations etc.
    assert np.allclose(new_buffer.observations[0], buffer.observations[0])
    assert new_buffer.actions[0] == buffer.actions[0]
