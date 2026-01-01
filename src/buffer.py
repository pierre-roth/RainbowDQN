import numpy as np
import torch
from pathlib import Path
from .config import Config

class SumTree:
    def __init__(self, capacity: int):
        """
        Initialize SumTree.
        
        Args:
            capacity (int): Number of leaf nodes (priorities).
        """
        self.capacity = capacity
        # Tree array stores sum of priorities
        # Size 2 * capacity - 1
        # Indices 0..capacity-2 are internal nodes
        # Indices capacity-1..2*capacity-2 are leaves (data)
        self.tree = np.zeros(2 * capacity - 1)
        self.data_pointer = 0
        self.size = 0

    def save(self, path: Path | str) -> None:
        np.savez_compressed(
            path,
            tree=self.tree,
            data_pointer=self.data_pointer,
            size=self.size
        )

    def load(self, path: Path | str) -> None:
        data = np.load(path)
        self.tree = data['tree']
        self.data_pointer = int(data['data_pointer'])
        self.size = int(data['size'])

    def add(self, priority: float) -> None:
        tree_idx = self.data_pointer + self.capacity - 1
        self.update(tree_idx, priority)
        
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1
            
    def update(self, tree_idx: int, priority: float) -> None:
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        
        # Propagate change
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v: float) -> tuple[int, float, int]:
        """
        Walk down the tree to find the leaf for value v.
        """
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            
            # If we reach bottom, end the search
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
                
            if v <= self.tree[left_child_idx]:
                parent_idx = left_child_idx
            else:
                v -= self.tree[left_child_idx]
                parent_idx = right_child_idx
                
        data_idx = leaf_idx - (self.capacity - 1)
        return leaf_idx, self.tree[leaf_idx], data_idx

    @property
    def total_priority(self) -> float:
        return self.tree[0]


from collections import deque

class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, observation_shape: tuple, alpha: float = Config.PER_ALPHA, n_step: int = Config.N_STEP, gamma: float = Config.GAMMA, device: str = Config.DEVICE):
        """
        Prioritized Replay Buffer with N-step Returns and Efficient Storage.
        Stores single frames to save memory and reconstructs stacks on sampling.
        """
        self.capacity = capacity
        self.alpha = alpha
        self.n_step = n_step
        self.gamma = gamma
        self.device = device
        self.stack_size = observation_shape[0] # e.g. 4
        
        self.tree = SumTree(capacity)
        # Fix: maxlen=None to prevent auto-dropping old transitions before processing
        self.n_step_buffer = deque(maxlen=None)
        
        # storage - Store single frames (H, W)
        self.frame_height, self.frame_width = observation_shape[1], observation_shape[2]
        self.observations = np.zeros((capacity, self.frame_height, self.frame_width), dtype=np.uint8)
        
        # Cursor for observations (runs ahead of tree pointer)
        self.obs_cursor = 0
        
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        # N-step done (used for TD target)
        self.dones = np.zeros(capacity, dtype=bool) 
        # Raw episode end (used for frame stacking)
        self.episode_ends = np.zeros(capacity, dtype=bool)

    def add(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool) -> None:
        """
        Add a new experience to the buffer. Auto-handles N-step buffering.
        
        Args:
            obs: Stacked observation (C, H, W).
            action: Action.
            reward: Reward.
            next_obs: Ignored (implied by sequence).
            done: Done flag.
        """
        # 1. Store Frame and Raw Done IMMEDIATELY
        # We store obs[-1] at the current cursor
        idx = self.obs_cursor
        self.observations[idx] = obs[-1]
        self.episode_ends[idx] = done # Store raw done for stacking logic
        self.obs_cursor = (self.obs_cursor + 1) % self.capacity
        
        # 2. Add metadata to n-step buffer
        # We store 'idx' to verify sync, though strict fifo should handle it
        self.n_step_buffer.append((idx, action, reward, done))
        
        # 3. Flush if done
        if done:
            while self.n_step_buffer:
                idx, a, r, d = self.n_step_buffer[0]
                
                # Compute Return
                R = 0
                final_d = False
                # Just process all remaining in buffer (flush)
                for i, transition in enumerate(self.n_step_buffer):
                     t_r = transition[2]
                     t_d = transition[3]
                     R += (self.gamma ** i) * t_r
                     if t_d:
                         final_d = True
                         break
                # If we flushed and didn't hit done, the last state is terminal?
                # No, if flushed via done, then SOMEWHERE in the buffer there is a done.
                # Actually, step 2 (flush) is ONLY called if `done` is True.
                # The `done` applies to the transition we just added.
                # So the LAST item in n_step_buffer has `t_d=True`.
                # So the loop is guaranteed to hit `final_d=True` eventually.
                # So this logic is correct for flush.

                self._store(idx, a, R, final_d)
                self.n_step_buffer.popleft()
            
            self.n_step_buffer.clear()
            return

        if len(self.n_step_buffer) < self.n_step + 1:
            return
            
        # 4. Get n-step transition
        idx, a, r, d = self.n_step_buffer[0]
        
        R = 0
        final_d = False
        # Iterate N steps
        for i in range(self.n_step):
            transition = self.n_step_buffer[i]
            t_r = transition[2]
            t_d = transition[3]
            R += (self.gamma ** i) * t_r
            if t_d:
                final_d = True
                break
        
        # 5. Store
        self._store(idx, a, R, final_d)
        
        # 6. Pop
        self.n_step_buffer.popleft()
            
    def _store(self, idx, action, reward, done):
        # Validate sync: The tree pointer MUST match the index where we stored the frame n steps ago
        # (or recently if flushing)
        # However, due to circular buffer, they should match modulo capacity.
        # assert idx == self.tree.data_pointer, f"Sync Error: ObsIdx {idx} != TreeIdx {self.tree.data_pointer}"
        
        # Initial priority
        max_priority = np.max(self.tree.tree[-self.capacity:]) if self.tree.size > 0 else 1.0
        if max_priority == 0: max_priority = 1.0
            
        ptr = self.tree.data_pointer
        
        # Metadata is stored at ptr (which should equal idx)
        self.actions[ptr] = action
        self.rewards[ptr] = reward
        self.dones[ptr] = done
        
        self.tree.add(max_priority)

    def _get_stacked_observation(self, idx: int) -> np.ndarray:
        """Reconstruct stacked observation from buffer."""
        # We need frames at idx-3, idx-2, idx-1, idx
        indices = [(idx - i) % self.capacity for i in range(self.stack_size - 1, -1, -1)]
        
        # Check for episode boundaries
        # If dones[k] is True, then k was terminal. 
        # So frames k+1... must not use frames <= k.
        # We iterate from right to left (idx-1 down to idx-3)
        # If we hit a done, we zero out everything before it.
        
        start_zero_idx = 0 # How many frames from start to zero out
        
        # indices are [t-3, t-2, t-1, t] (if stack=4)
        # We check dones at t-1, t-2, t-3.
        # If dones[t-1] is True, then t is start of new episode. t-1, t-2, t-3 must be zeroed.
        # Logic: Check last stack_size-1 frames.
        
        for i in range(1, self.stack_size):
            # Check frame at index corresponding to loop `i` steps back
            # indices[-1] is current (idx). indices[-1-i] is `i` steps back.
            prev_idx = indices[-1 - i]
            # If prev_idx was done, then it terminates the history for everything AFTER it.
            # But wait, indices are ordered.
            if self.episode_ends[prev_idx]:
                # Found a done at `prev_idx`.
                # This means `prev_idx` is the END of an episode.
                # The frame at `prev_idx+1` (which is `indices[-i]`) starts a new one.
                # So we must zero out everything including and before `prev_idx`.
                # In our `indices` list, that corresponds to `indices[:-i]`.
                start_zero_idx = self.stack_size - i
                break

            # Boundary Check:
            # If prev_idx is the *most recently written* frame (obs_cursor - 1),
            # and the current frame (indices[-i]) is the *oldest* frame (obs_cursor),
            # then we have crossed the buffer boundary (Newest -> Oldest).
            # This history is invalid.
            current_head = (self.obs_cursor - 1 + self.capacity) % self.capacity
            if prev_idx == current_head:
                 start_zero_idx = self.stack_size - i
                 break
                
        frames = self.observations[indices] # (4, 84, 84)
        if start_zero_idx > 0:
            frames[:start_zero_idx] = 0
            
        return frames

    def sample(self, batch_size: int, beta: float = 0.4) -> tuple[tuple, torch.Tensor, list[int]]:
        """
        Sample a batch of experiences.
        
        Args:
            batch_size (int): Size of batch.
            beta (float, optional): Importance sampling exponent. Defaults to 0.4.

        Returns:
            tuple: ((obs, act, rew, next_obs, done), weights, indices)
        """
        weights = []
        indices = []
        
        # 1. Sample Indices
        segment = self.tree.total_priority / batch_size
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            idx, p, data_idx = self.tree.get_leaf(s)
            
            indices.append(idx)
            
            prob = p / self.tree.total_priority
            weight = (self.tree.size * prob) ** (-beta)
            weights.append(weight)
            
        # 2. Reconstruct Data
        # indices is tree indices. data_indices maps to buffer
        data_indices = np.array(indices) - (self.tree.capacity - 1)
        
        obs_batch = np.zeros((batch_size, self.stack_size, self.frame_height, self.frame_width), dtype=np.uint8)
        next_obs_batch = np.zeros((batch_size, self.stack_size, self.frame_height, self.frame_width), dtype=np.uint8)
        
        for i, idx in enumerate(data_indices):
            obs_batch[i] = self._get_stacked_observation(idx)
            
            # Next Observation
            # If current transition is Done, next_obs doesn't matter (masked by loss).
            # If Not Done, next_obs is at idx + n_step.
            # We assume Buffer is large enough that idx+n_step hasn't been overwritten if idx hasn't.
            next_idx = (idx + self.n_step) % self.capacity
            next_obs_batch[i] = self._get_stacked_observation(next_idx)

        act_batch = self.actions[data_indices]
        rew_batch = self.rewards[data_indices]
        done_batch = self.dones[data_indices]
            
        # Normalize weights
        weights = np.array(weights)
        weights /= weights.max()
        
        # Convert to tensors
        obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device) / 255.0
        act_t = torch.as_tensor(act_batch, dtype=torch.int64, device=self.device)
        rew_t = torch.as_tensor(rew_batch, dtype=torch.float32, device=self.device)
        next_obs_t = torch.as_tensor(next_obs_batch, dtype=torch.float32, device=self.device) / 255.0
        done_t = torch.as_tensor(done_batch, dtype=torch.float32, device=self.device)
        weights_t = torch.as_tensor(weights, dtype=torch.float32, device=self.device)
        
        return (obs_t, act_t, rew_t, next_obs_t, done_t), weights_t, indices

    def update_priorities(self, tree_indices: list[int], errors: np.ndarray) -> None:
        """
        Update priorities of sampled transitions.
        
        Args:
            tree_indices (list[int]): Indices in the tree.
            errors (np.ndarray): TD-errors (used to calculate priority).
        """
        for idx, error in zip(tree_indices, errors):
            priority = (error + 1e-6) ** self.alpha
            self.tree.update(idx, priority)
            
    def __len__(self):
        return self.tree.size
        
    def save(self, path: Path | str) -> None:
        np.savez_compressed(
            path,
            observations=self.observations,
            actions=self.actions,
            rewards=self.rewards,
            dones=self.dones,
            episode_ends=self.episode_ends,
            tree_tree=self.tree.tree,
            tree_pointer=self.tree.data_pointer,
            tree_size=self.tree.size,
            capacity=self.capacity,
            obs_cursor=self.obs_cursor
        )

    def load(self, path: Path | str) -> None:
        try:
            data = np.load(path)
            
            saved_capacity = int(data.get('capacity', self.capacity)) 
            if saved_capacity != self.capacity:
                print(f"Warning: Buffer capacity mismatch! Loaded: {saved_capacity}, Current: {self.capacity}.")

            self.observations = data['observations']
            self.actions = data['actions']
            self.rewards = data['rewards']
            self.dones = data['dones']
            self.episode_ends = data.get('episode_ends', np.zeros_like(self.dones))
            
            self.tree.tree = data['tree_tree']
            self.tree.data_pointer = int(data['tree_pointer'])
            self.tree.size = int(data['tree_size'])
            
            # CRITICAL FIX:
            # On resume, the n_step_buffer is lost (it's volatile).
            # If obs_cursor represents the head of the frame stream, and data_pointer represents the head of the metadata stream,
            # efficiently they are offset by n_step.
            # Since we lost the 'pending' metadata in n_step_buffer, we MUST reset the frame cursor
            # to match the metadata pointer. This effectively 'rewinds' the frame storage to where
            # the last valid committed metadata was.
            # If we don't do this, new metadata (actions) will be written to the data_pointer location,
            # but they will be associated with the OLD frames that were "pending" (stored ahead of data_pointer).
            self.obs_cursor = self.tree.data_pointer
            
            print(f"Loaded buffer with {self.tree.size} transitions. Synced cursor to {self.obs_cursor}.")
        except Exception as e:
            print(f"Failed to load buffer from {path}: {e}")
            raise e
