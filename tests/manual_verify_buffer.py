import numpy as np
from src.buffer import PrioritizedReplayBuffer

def test_stacking_logic():
    print("Testing Buffer Stacking Logic...")
    
    # Setup small buffer
    # Shape: (1, 1, 1) for simplicity (1 pixel frames)
    # Stack: 4
    # Cap: 20
    buffer = PrioritizedReplayBuffer(capacity=20, observation_shape=(4, 1, 1), n_step=1)
    
    # We will verify _get_stacked_observation directly using the private method logic
    # (or we can expose it, but let's just inspect the buffer state or add a helper if needed)
    # Accessing protected member for testing is acceptable here.

    # Episode 1: Frames 1, 2, 3, 4, 5 (Done at 5)
    # Episode 2: Frames 6, 7, 8 (Done at 8)
    
    # Add Episode 1
    # obs=Frame 1
    buffer.add(np.full((4,1,1), 1), 0, 0, np.zeros((4,1,1)), False) # Cursor 0 -> Frame 1
    buffer.add(np.full((4,1,1), 2), 0, 0, np.zeros((4,1,1)), False) # Cursor 1 -> Frame 2
    buffer.add(np.full((4,1,1), 3), 0, 0, np.zeros((4,1,1)), False) # Cursor 2 -> Frame 3
    buffer.add(np.full((4,1,1), 4), 0, 0, np.zeros((4,1,1)), False) # Cursor 3 -> Frame 4
    buffer.add(np.full((4,1,1), 5), 0, 0, np.zeros((4,1,1)), True)  # Cursor 4 -> Frame 5 (DONE)

    # Add Episode 2
    buffer.add(np.full((4,1,1), 6), 0, 0, np.zeros((4,1,1)), False) # Cursor 5 -> Frame 6
    buffer.add(np.full((4,1,1), 7), 0, 0, np.zeros((4,1,1)), False) # Cursor 6 -> Frame 7
    buffer.add(np.full((4,1,1), 8), 0, 0, np.zeros((4,1,1)), True)  # Cursor 7 -> Frame 8 (DONE)

    # Helper to print stack
    def check_idx(idx, expected_frames, description):
        stack = buffer._get_stacked_observation(idx)
        # stack is (4, 1, 1). Flatten to (4,)
        flat = stack.flatten()
        print(f"Index {idx} ({description}): Expected {expected_frames} -> Got {flat}")
        if np.array_equal(flat, expected_frames):
             print("  [PASS]")
        else:
             print("  [FAIL]")

    print("\n--- Episode 1 Checks ---")
    # Idx 0 (Frame 1): Should be [0, 0, 0, 1] (Stack 4)
    # Logic: idx-1 (wrap 19) is empty/0? Actually buffer inits to 0.
    # But crucially, start of episode logic relies on 'dones'.
    # Since we just started, previous dones are 0.
    # Wait, 'episode_ends' are 0.
    # So it will read wrap-around 0s. Which is technically correct for clean buffer.
    check_idx(0, [0, 0, 0, 1], "Start of Ep 1 (Frame 1)")
    
    # Idx 2 (Frame 3): Should be [0, 1, 2, 3]
    check_idx(2, [0, 1, 2, 3], "Mid Ep 1 (Frame 3)")
    
    # Idx 3 (Frame 4): Should be [1, 2, 3, 4]
    check_idx(3, [1, 2, 3, 4], "Mid Ep 1 (Frame 4)")
    
    # Idx 4 (Frame 5): Should be [2, 3, 4, 5]
    check_idx(4, [2, 3, 4, 5], "End Ep 1 (Frame 5)")

    print("\n--- Episode 2 Checks ---")
    # Idx 5 (Frame 6): Start of Ep 2.
    # Previous frame (Idx 4) was DONE.
    # So history should be wiped. 
    # expected: [0, 0, 0, 6]
    check_idx(5, [0, 0, 0, 6], "Start of Ep 2 (Frame 6)")
    
    # Idx 6 (Frame 7): [0, 0, 6, 7]
    # idx-1 (5) is not done.
    # idx-2 (4) IS done. So wipe before 4? No, wipe history beyond 5.
    # Loop: 
    # i=1: prev=5. done[5]? No.
    # i=2: prev=4. done[4]? Yes (End of Ep 1).
    # So zero out indices[:-2] -> frames[:2] = 0.
    # frames: [0, 0, 6, 7]
    check_idx(6, [0, 0, 6, 7], "2nd Frame of Ep 2")
    
    print("\n--- Wrap Around Check ---")
    # Fill until wrap. Cap is 20.
    # Currently at cursor 8.
    # Fill 9..19.
    for i in range(9, 20):
        buffer.add(np.full((4,1,1), i+1), 0, 0, np.zeros((4,1,1)), False)
        
    # Now add frame at index 0 (Val 21).
    buffer.add(np.full((4,1,1), 21), 0, 0, np.zeros((4,1,1)), False) 
    
    # Check Idx 0 (Val 1 - Oldest).
    # Previous frame is 21 (Newest). 
    # Logic should detect we crossed obs_cursor and zero out history.
    # Expected: [0, 0, 0, 1]
    check_idx(0, [0, 0, 0, 1], "Wrap Around (Idx 0 - Oldest)")

if __name__ == "__main__":
    test_stacking_logic()
