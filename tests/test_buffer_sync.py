
def test_buffer_resume_sync(tmp_path):
    """Regression test for N-step buffer sync bug on resume."""
    import numpy as np
    from src.buffer import PrioritizedReplayBuffer
    shape = (4, 5, 5)
    # n_step=3, capacity=100
    buffer = PrioritizedReplayBuffer(capacity=100, observation_shape=shape, n_step=3)
    
    # 1. Fill buffer with some data (5 items)
    # Lag is 3. So items 0, 1 are finalized. 
    # Items 2, 3, 4 are pending in n_step_buffer.
    # obs_cursor = 5. tree.data_pointer = 2.
    for i in range(5):
        obs = np.full(shape, i, dtype=np.uint8)
        buffer.add(obs, i, 1.0, obs, False)
        
    assert buffer.obs_cursor == 5
    assert buffer.tree.data_pointer == 2
    
    # 2. Save
    save_path = tmp_path / "buffer_sync.npz"
    buffer.save(save_path)
    
    # 3. Load
    new_buffer = PrioritizedReplayBuffer(capacity=100, observation_shape=shape, n_step=3)
    new_buffer.load(save_path)
    
    # 4. Verify Sync
    # The load function should have reset obs_cursor to tree.data_pointer (2)
    # to account for lost n-step buffer
    assert new_buffer.obs_cursor == new_buffer.tree.data_pointer
    assert new_buffer.obs_cursor == 2
    
    # 5. Add new data and verify alignment
    # If we add data now, it should write to index 2 (after n_step delay)
    # matching the frame we are about to overwrite at index 2.
    # We add 4 items to flush index 2.
    for i in range(5, 9): # 5, 6, 7, 8
        obs = np.full(shape, i, dtype=np.uint8)
        new_buffer.add(obs, 100+i, 1.0, obs, False)
        
    # Sample index 2
    # We need to find the batch item that corresponds to tree leaf 2
    # We can cheat and just look at the internal storage since this is a test
    idx = 2
    action = new_buffer.actions[idx]
    # buffer.observations stores single frames (H, W), so only 2 dims
    frame_val = new_buffer.observations[idx][0, 0]
    
    # Expect action from step 5 (100+5 = 105)
    # Expect frame from step 5 (val=5)
    assert action == 105
    assert frame_val == 5
